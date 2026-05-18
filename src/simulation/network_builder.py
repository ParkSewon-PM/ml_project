import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def _resolve_sumo_tool(name: str) -> str:
    """Resolve a SUMO tool (netconvert etc.) using SUMO_HOME if needed."""
    if shutil.which(name):
        return name
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        full = os.path.join(sumo_home, "bin", name)
        if os.path.isfile(full):
            return full
        full_exe = full + ".exe"
        if os.path.isfile(full_exe):
            return full_exe
    raise FileNotFoundError(
        f"'{name}' not found in PATH or SUMO_HOME={sumo_home!r}. Install SUMO and set SUMO_HOME."
    )


def build_single_link_network(
    output_path: str | Path,
    link_length: float = 5000.0,
    num_lanes: int = 2,
    speed_limit: float = 33.33,
) -> Path:
    """Create a minimal SUMO network with one edge (link).

    The network has two nodes connected by a single edge.
    Default link_length is 5000m (5km).
    Generates .nod.xml, .edg.xml, and runs netconvert to produce .net.xml.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Nodes
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="start", x="0.0", y="0.0", type="priority")
    ET.SubElement(
        nodes,
        "node",
        id="end",
        x=str(link_length),
        y="0.0",
        type="priority",
    )

    nodes_path = output_path.with_suffix(".nod.xml")
    ET.ElementTree(nodes).write(str(nodes_path), xml_declaration=True, encoding="UTF-8")

    # Edges
    edges = ET.Element("edges")
    ET.SubElement(
        edges,
        "edge",
        id="link0",
        **{"from": "start", "to": "end"},
        numLanes=str(num_lanes),
        speed=str(speed_limit),
    )

    edges_path = output_path.with_suffix(".edg.xml")
    ET.ElementTree(edges).write(str(edges_path), xml_declaration=True, encoding="UTF-8")

    # Run netconvert to produce .net.xml
    net_path = output_path.parent / "network.net.xml"
    netconvert_bin = _resolve_sumo_tool("netconvert")
    result = subprocess.run(
        [
            netconvert_bin,
            "--node-files",
            str(nodes_path),
            "--edge-files",
            str(edges_path),
            "--output-file",
            str(net_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed (exit {result.returncode}): {result.stderr.strip()}")

    return net_path


def _add_vtype(
    parent: ET.Element,
    vtype_id: str,
    params: dict[str, float],
    max_speed: float = 33.33,
) -> None:
    """Add a vType element with the given parameters."""
    attrs: dict[str, str] = {"id": vtype_id}
    for key in ("length", "tau", "decel", "minGap", "sigma", "accel"):
        if key in params:
            attrs[key] = str(round(params[key], 4))
    if "speedFactor" in params:
        attrs["speedFactor"] = str(round(params["speedFactor"], 4))
    attrs["maxSpeed"] = str(round(max_speed, 4))
    ET.SubElement(parent, "vType", **attrs)


def build_bottleneck_network(
    output_path: str | Path,
    link_length: float = 5000.0,
    num_lanes: int = 3,
    bottleneck_lanes: int | None = None,
    bottleneck_length: float = 500.0,
    speed_limit: float = 33.33,
    bottleneck_speed: float | None = None,
) -> Path:
    """Create a SUMO network with a lane-drop bottleneck.

    Layout::

        [start] --link0 (num_lanes)--> [mid] --link1 (bottleneck_lanes)--> [end]

    Measurement and probe extraction happen on ``link0`` only.
    ``link1`` exists solely to create queue spillback into ``link0``.

    Parameters
    ----------
    bottleneck_lanes:
        Number of lanes on the downstream link.  Defaults to
        ``num_lanes - 1`` (single lane drop).
    bottleneck_length:
        Length of the bottleneck link in meters.  Only needs to be long
        enough to sustain the capacity drop; 500 m is usually sufficient.
    bottleneck_speed:
        Speed limit on the bottleneck link (m/s).  Defaults to same as
        ``speed_limit``.  Set lower to reduce bottleneck throughput and
        force higher density on the measurement link.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if bottleneck_lanes is None:
        bottleneck_lanes = max(1, num_lanes - 1)
    if bottleneck_speed is None:
        bottleneck_speed = speed_limit

    # Nodes
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="start", x="0.0", y="0.0", type="priority")
    ET.SubElement(nodes, "node", id="mid", x=str(link_length), y="0.0", type="priority")
    ET.SubElement(
        nodes,
        "node",
        id="end",
        x=str(link_length + bottleneck_length),
        y="0.0",
        type="priority",
    )

    nodes_path = output_path.with_suffix(".nod.xml")
    ET.ElementTree(nodes).write(str(nodes_path), xml_declaration=True, encoding="UTF-8")

    # Edges
    edges = ET.Element("edges")
    ET.SubElement(
        edges,
        "edge",
        id="link0",
        **{"from": "start", "to": "mid"},
        numLanes=str(num_lanes),
        speed=str(speed_limit),
    )
    ET.SubElement(
        edges,
        "edge",
        id="link1",
        **{"from": "mid", "to": "end"},
        numLanes=str(bottleneck_lanes),
        speed=str(bottleneck_speed),
    )

    edges_path = output_path.with_suffix(".edg.xml")
    ET.ElementTree(edges).write(str(edges_path), xml_declaration=True, encoding="UTF-8")

    # Run netconvert
    net_path = output_path.parent / "network.net.xml"
    netconvert_bin = _resolve_sumo_tool("netconvert")
    result = subprocess.run(
        [
            netconvert_bin,
            "--node-files",
            str(nodes_path),
            "--edge-files",
            str(edges_path),
            "--output-file",
            str(net_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed (exit {result.returncode}): {result.stderr.strip()}")

    return net_path


def build_route_file(
    output_path: str | Path,
    demand_vehph: int = 1000,
    begin: float = 0.0,
    end: float = 3600.0,
    truck_ratio: float = 0.0,
    passenger_params: dict[str, float] | None = None,
    truck_params: dict[str, float] | None = None,
    speed_limit: float = 33.33,
    bottleneck: bool = False,
) -> Path:
    """Create a SUMO route file with passenger/truck flows.

    Args:
        demand_vehph: Total demand (veh/hr).
        truck_ratio: Fraction of trucks in total demand.
        passenger_params: Vehicle type params for passenger cars.
        truck_params: Vehicle type params for trucks.
        speed_limit: Speed limit (m/s) for maxSpeed.
        bottleneck: If True, route spans "link0 link1" (for bottleneck network).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    routes = ET.Element("routes")

    # Vehicle types
    pass_p = passenger_params or {"length": 4.5}
    _add_vtype(routes, "passenger", pass_p, max_speed=speed_limit)

    if truck_ratio > 0 and truck_params:
        _add_vtype(routes, "truck", truck_params, max_speed=speed_limit)

    # Route
    route_edges = "link0 link1" if bottleneck else "link0"
    ET.SubElement(routes, "route", id="r0", edges=route_edges)

    # Passenger flow
    passenger_demand = max(1, int(demand_vehph * (1 - truck_ratio)))
    ET.SubElement(
        routes,
        "flow",
        id="flow_passenger",
        route="r0",
        type="passenger",
        begin=str(begin),
        end=str(end),
        vehsPerHour=str(passenger_demand),
        departLane="best",
        departSpeed="max",
    )

    # Truck flow
    if truck_ratio > 0 and truck_params:
        truck_demand = max(1, int(demand_vehph * truck_ratio))
        ET.SubElement(
            routes,
            "flow",
            id="flow_truck",
            route="r0",
            type="truck",
            begin=str(begin),
            end=str(end),
            vehsPerHour=str(truck_demand),
            departLane="best",
            departSpeed="max",
        )

    ET.ElementTree(routes).write(str(output_path), xml_declaration=True, encoding="UTF-8")
    return output_path
