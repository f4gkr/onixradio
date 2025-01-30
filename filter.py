from onnx import numpy_helper,TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
import numpy as np

cpxcount  = 8
hlen = 4 
floatcount = cpxcount * 2


taps = make_tensor_value_info('taps', TensorProto.FLOAT, [hlen])
input = make_tensor_value_info('input', TensorProto.FLOAT, [floatcount]) 

out = make_tensor_value_info('out', TensorProto.FLOAT, [2*(cpxcount-hlen+1)])  

#-------------------------------------------------------------------
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
                inputs=["input", "starts", "ends", "axes", "steps" ],
                outputs= ['reals'] )

new_shape = np.array([1,1,1,cpxcount],dtype=np.int32) 
new_shape = numpy_helper.from_array(new_shape, name='new_shape')
rsh1 = make_node('Reshape',
                    inputs=['reals','new_shape'],
                    outputs=['rreals'])

lp1 = make_node('Conv',
                inputs=['rreals','taps'],
                outputs= ['hreals'],
                kernel_shape=[1,hlen],
                pads=[0, 0, 0, 0],
                auto_pad='NOTSET')

flat_shape = np.array([cpxcount-hlen+1],dtype=np.int32) 
flat_shape = numpy_helper.from_array(flat_shape, name='flat')
r2 = make_node('Reshape',
                    inputs=['hreals','flat'],
                    outputs=['hrealsflat'])

#-------------------------------------------------------------------
vistarts = np.zeros( 1 ,dtype=np.int64) 
vistarts[0] = 1
istarts = numpy_helper.from_array(vistarts, name='istarts')

selimags = make_node('Slice',
                inputs=["input", "istarts", "ends", "axes", "steps" ],
                outputs= ['imags'] )

rsh2 = make_node('Reshape',
                    inputs=['imags','new_shape'],
                    outputs=['rimags'])

lp2 = make_node('Conv',
                inputs=['rimags','taps'],
                outputs= ['himags'],
                kernel_shape=[1,hlen],
                pads=[0, 0, 0, 0],
                auto_pad='NOTSET')

r3 = make_node('Reshape',
                    inputs=['himags','flat'],
                    outputs=['himagsflat'])
#-------------------------------------------------------------------
concat = make_node('Concat',
                   inputs=['hrealsflat','himagsflat'],
                   outputs=['IandQ'],
                   axis=0)

# generate sequence to alternate IQ while we have IIII...IIIQQQQ.....Q
nouts = cpxcount - hlen + 1
vSelector = np.zeros( 2*nouts ) 
for x in range(0,nouts):
    vSelector[2*x  ] = x 
    vSelector[2*x+1] = nouts+x 

print( vSelector )

selector = numpy_helper.from_array(vSelector, name='selector')
assemble = make_node('GatherElements', 
                  inputs=['IandQ', 'selector'],
                  outputs= ['out'],
                  axis=0 )

#-------------------------------------------------------------------

graph = make_graph(nodes=[selreals , selimags, rsh1, rsh2, lp1, lp2, r2, r3, concat, assemble], 
                   name='fir', 
                   inputs=[taps,input], 
                   outputs=[out ] ,
                   initializer=[starts,istarts,ends, axes, steps,selector,new_shape,flat_shape])

onnx_model = make_model(graph)
check_model(onnx_model)

with open("filter.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = ReferenceEvaluator(onnx_model)

taps =np.array(
    [
        [
            [
                [.25 , .25 , .25 , .25 ]
            ]
        ]        
    ], dtype=np.float32)
vA =np.array( [1,10, 1,10, 1,10, 1,10, 1,10, 1,10, 1,10, 1,10 ], dtype=np.float32) 
feeds = {'taps': taps, 'input': vA }
print(sess.run(None, feeds))