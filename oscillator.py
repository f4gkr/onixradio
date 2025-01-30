from onnx import numpy_helper,TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
import numpy as np

cpxcount  = 16384
floatcount = cpxcount * 2

alpha = make_tensor_value_info('alpha', TensorProto.FLOAT, [1])
out = make_tensor_value_info('out', TensorProto.FLOAT, [floatcount]) 

incrementer = np.zeros( floatcount ,dtype=np.float32) 
for x in range(0,cpxcount):
    incrementer[2*x] = x  
    incrementer[2*x+1] = x 

vincrementer  = numpy_helper.from_array(incrementer, name='incrementer')
new_shape = np.array([1,floatcount],dtype=np.int32)
new_shape = numpy_helper.from_array(new_shape, name='new_shape')

vselreal = np.zeros( floatcount )
vselreal[0::2] = 1
selreal  = numpy_helper.from_array(vselreal, name='selreal')

rep = make_node('Expand',
                inputs=['alpha','new_shape'],
                outputs= ['repalpha'])

m1 = make_node('Mul',
               inputs = ['repalpha','incrementer'],
                outputs= ['alphat'])

cos = make_node('Cos',
                inputs = ['alphat'],
                outputs= ['cosalpha'])

sin = make_node('Sin',
                inputs = ['alphat'],
                outputs= ['sinalpha'])

cpx = make_node('Where',
                 inputs=['selreal','cosalpha','sinalpha'],
                 outputs=['out'])


graph = make_graph(nodes=[rep,m1,cos,sin,cpx], 
                   name='oscillator', 
                   inputs=[alpha], 
                   outputs=[out] ,
                   initializer=[vincrementer,new_shape, selreal])

onnx_model = make_model(graph)
check_model(onnx_model)

# The serialization
with open("oscillator.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = ReferenceEvaluator(onnx_model)
vA =np.array([0.1256637061435927], dtype=np.float32) 


feeds = {'alpha': vA }

result = sess.run(None, feeds)
oscillator = result[0]
print(oscillator)
oscillator.tofile('floats.cf32')