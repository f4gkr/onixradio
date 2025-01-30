from onnx import numpy_helper,TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
import numpy as np

cpxcount  = 8 
floatcount = cpxcount * 2

# generate sequence (0,1,3,2,5,4,....)
vSelector = np.zeros( floatcount )
vSelector[0] = 1 
vSelector[1] = 0
for x in range(2,floatcount):
    vSelector[x] = vSelector[x-2]+2

vselreal = np.zeros( floatcount )
vselreal[0::2] = 1
selector = numpy_helper.from_array(vSelector, name='selector')
selreal  = numpy_helper.from_array(vselreal, name='selreal') 

input = make_tensor_value_info('input', TensorProto.FLOAT, [floatcount]) 
out = make_tensor_value_info('out', TensorProto.FLOAT, [cpxcount ]) 

m1 = make_node('Mul',
                inputs=['input','input'],
                outputs= ['modulus'])

s1 = make_node('GatherElements', 
                  inputs=['modulus', 'selector'],
                  outputs= ['sMod'],
                  axis=0 )

i2q2 = make_node('Add',
                  inputs=['modulus','sMod'],
                  outputs= ['i2q2'])

vstarts = np.zeros( 1 ,dtype=np.int64) 
vstarts[0] = 0
starts = numpy_helper.from_array(vstarts, name='starts')
 
vends = np.zeros( 1 ,dtype=np.int64) 
vends[0] = floatcount
ends = numpy_helper.from_array(vends, name='ends')

vaxes = np.zeros( 1 ,dtype=np.int64) 
vaxes[0] = 0
axes = numpy_helper.from_array(vaxes, name='axes')

vsteps = np.zeros( 1 ,dtype=np.int64) 
vsteps[0] = 2
steps = numpy_helper.from_array(vsteps, name='steps')

selreals = make_node('Slice',
                inputs=["i2q2", "starts", "ends", "axes", "steps" ],
                outputs= ['oneoftwo'] )

sqrt1 = make_node('Sqrt',
                inputs=['oneoftwo' ],
                outputs= ['out'])

graph = make_graph(nodes=[m1 , s1, i2q2, selreals, sqrt1], 
                   name='AMDemod', 
                   inputs=[input], 
                   outputs=[out ] ,
                   initializer=[starts,ends, axes, steps,selector ])

onnx_model = make_model(graph)
check_model(onnx_model)

with open("amdemod.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = ReferenceEvaluator(onnx_model)
# basic tests
vA =np.array( [1,10, 1,10, 1,10, 1,10, 1,10, 1,10, 1,10, 1,10 ], dtype=np.float32) 
feeds = { 'input': vA }
print(sess.run(None, feeds))