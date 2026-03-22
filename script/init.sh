# caution, you need to source this script instead of execute it
# disable sfploadmacro error
# export TT_METAL_DISABLE_SFPLOADMACRO=1
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/tt/tt-metal  # for docker container tt-metal
# https://github.com/tenstorrent/ttsim/releases
export TT_METAL_SIMULATOR=/tt/sim/libttsim_wh.so
export TT_METAL_DPRINT_CORES=0,0
# export TT_METAL_DPRINT_RISCVS=TR0
# export TT_METAL_DPRINT_CORES='(0,0)-(8,0)'  # a row
