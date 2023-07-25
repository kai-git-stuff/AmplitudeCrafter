from jitter.amplitudes.dalitz_plot_function import DalitzDecayFunctions, chain
from jitter.constants import spin as sp
from AmplitudeCrafter.Resonances import map_arguments
from jax import numpy as jnp
from jax import jit
from jitter.fitting import FitParameter

def run_lineshape(resonance_tuple,args,mapping_dict,bls_in,bls_out,masses):
    """
    Performs the execution of a single Resonance lineshape
    L and L_0 values are set manually
    The resonance tuple includes information no longer needed
    For legacy resoans it is still carried, but will be droppend in future versions
    """
    s,p,hel,lineshape_func,channel = resonance_tuple
    lineshape = {}
    i,j,k = {1:[2,3,1], 2:[3,1,2], 3:[1,2,3]}[channel]
    mapping_dict["M0"] = masses[0]
    mapping_dict["MI"] = masses[i]
    mapping_dict["MJ"] = masses[j]
    mapping_dict["MK"] = masses[k]
    for LS,b in bls_out.items():
        for LS_i,b_i in bls_in.items():
            L,S = LS
            L_0, S_0 = LS_i
            mapping_dict["L"] = L  # set the correct angular momentum
            mapping_dict["L_0"] = L_0 # set the correct angular momentum
            lineshape[(L_0,L)] = lineshape_func(*map_arguments(args,mapping_dict=mapping_dict))

    return (s,hel,lineshape)

def construct_function(masses,spins,parities,params,mapping_dict,
                        resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,phsp,
                        total_absolute=True, just_in_time_compile=True, numericArgs=True):
    """
    Function to construct the actual ampltude function from the Dalitz Amplitude
    The function will take all non fixed parameters as defined in the yml files

    total_absolute will descide weather the dalitz plot funciton will be returned or
    the sum over all helicity configurations
    """

    # we need to translate all _complex values into real and imaginary
    needed_params = params
    needed_names = [p.name for p in needed_params] # TODO: Names are wrong here!!! They do not reflect the names in the mapping dict!
    start = map_arguments(needed_params,numeric=numericArgs)
    mapping_dict_global = mapping_dict
    decay = DalitzDecayFunctions(*masses,*spins,*parities,phsp=phsp)

    def fill_args(smp,args,mapping_dict):
        """
        Put the arguments into the correct places in the mapping dict
        This has to be used, so no JAX Tracer objects leak
        Thus the parameter.update() functionlaity may not be used!
        Instead the parmeter value can be retrieved by using parameter(value_dict=mapping_dict)
        This will read the needed value from the dict, but still make use of the internal logic to construct
        complex numbers and adhere to naming schemes.
        """
        if len(args) == 1:
            # wierd bug...
            # need to investigate this
            args = args[0]
        dtc = mapping_dict.copy()
        for param_name, val in zip(needed_names,args):
            if isinstance(val,FitParameter):
                val = val()
            dtc[param_name] = val
        dtc["sigma3"] = decay["sigma2"](smp)
        dtc["sigma2"] = decay["sigma2"](smp)
        dtc["sigma1"] = decay["sigma1"](smp)
        return dtc

    resonances_filled = [[None for j,r in enumerate(res)] for i, res in enumerate(resonance_tuples)  ]

    def update(mapping_dict,bls_out):
        """
        We only need to update the resonance lineshapes, where parameters of the lineshape are free
        """
        for i,l in enumerate(resonances_filled):
            for j, _ in enumerate(l):
                resonances_filled[i][j] = run_lineshape(resonance_tuples[i][j],resonance_args[i][j],mapping_dict,bls_in[i][j],bls_out[i][j],masses)

    if total_absolute:
        def f(smp,args):
            mapping_dict = fill_args(smp,args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in,mapping_dict=mapping_dict)
            bls_out_mapped = map_arguments(bls_out,mapping_dict=mapping_dict)
            update(mapping_dict,bls_out_mapped)

            def O(nu,lambdas):       
                tmp = (chain(smp,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3,*masses,*spins,*parities) + 
                       chain(smp,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2,*masses,*spins,*parities) + 
                       chain(smp,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1,*masses,*spins,*parities)
                )
                return tmp
            ampl =  sum(
                sum(
                    jnp.abs(O(ld,[la,lb,lc]))**2  
                        for la,lb,lc in decay["HelicityOptions"]
                            ) for ld in sp.direction_options(decay["sd"]))
            return ampl
    else:
        def f(smp,args,nu,*lambdas):
            mapping_dict = fill_args(smp,args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in,mapping_dict=mapping_dict)
            bls_out_mapped = map_arguments(bls_out,mapping_dict=mapping_dict)
            update(mapping_dict,bls_out_mapped)

            def O(nu,lambdas):       
                tmp = (chain(smp,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3,*masses,*spins,*parities) + 
                       chain(smp,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2,*masses,*spins,*parities) + 
                       chain(smp,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1,*masses,*spins,*parities)
                )
                return tmp
            return O(nu,lambdas)
    if just_in_time_compile:
        f = jit(f)
    return f, start