#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  Parameters ('dtype' has 4 options: uint8, uint16, int32, float32)
# ------------------------------------------------------------------------------
n=60000
qn=100
d=50
c=2.0
leaf=4000
L=30
M=10
dtype=uint8
dname=Mnist
pf=../data/${dname}/${dname}

# ------------------------------------------------------------------------------
#  Running Scripts
# ------------------------------------------------------------------------------
p_list=(2.0) 
z_list=(0.0)
# p_list=(0.5 1.0 2.0)
# z_list=(1.0 0.0 0.0)
length=`expr ${#p_list[*]} - 1`

for j in $(seq 0 ${length})
do
  p=${p_list[j]}
  z=${z_list[j]}
  of=../results/${dname}/c=${c}_p=${p}/

  # ----------------------------------------------------------------------------
  #  Ground Truth
  # ----------------------------------------------------------------------------
  ./qalsh -alg 0 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf}

  # ----------------------------------------------------------------------------
  #  QALSH_Plus
  # ----------------------------------------------------------------------------
  ./qalsh -alg 1 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
    -lf ${leaf} -L ${L} -M ${M} -dt ${dtype} -pf ${pf} -of ${of}

  # ----------------------------------------------------------------------------
  #  QALSH
  # ----------------------------------------------------------------------------
  ./qalsh -alg 2 -n ${n} -qn ${qn} -d ${d} -p ${p} -z ${z} -c ${c} \
    -dt ${dtype} -pf ${pf} -of ${of}

  # ----------------------------------------------------------------------------
  #  Linear Scan
  # ----------------------------------------------------------------------------
  ./qalsh -alg 3 -n ${n} -qn ${qn} -d ${d} -p ${p} -dt ${dtype} -pf ${pf} \
    -of ${of}
done
