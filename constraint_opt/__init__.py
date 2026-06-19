from __future__ import annotations

from .factory import register_problem, resolve

# Register encodings and runners
from .mip_gurobi import build_mip_gurobi_encoding
from .cp_sat import build_cp_sat_encoding

register_problem("mip-gurobi", build_mip_gurobi_encoding)

register_problem("cp-sat", build_cp_sat_encoding)