from juliacall import Main as jl
from juliacall import Pkg as jlPkg

jlPkg.activate(".")
jl.seval("using Genqo")

from .genqo import *
from .sweep import *
