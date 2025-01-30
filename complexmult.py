from onnx import numpy_helper,TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator
from onnx import version_converter
import numpy

cpxcount  = 1024
floatcount = cpxcount * 2

# generate sequence (0,1,3,2,5,4,....)
vSelector = numpy.zeros( floatcount, dtype=numpy.int32)
vSelector[0] = 1 
vSelector[1] = 0
for x in range(2,floatcount):
    vSelector[x] = vSelector[x-2]+2

vselreal = numpy.zeros( floatcount , dtype=bool)
vselreal[0::2] = True
 
#vSelector = numpy.array([1,0,3,2],dtype=numpy.int32)
# selector = make_tensor_value_info('selector', TensorProto.INT32, [4])
selector = numpy_helper.from_array(vSelector, name='selector')
selreal  = numpy_helper.from_array(vselreal, name='selreal')

A = make_tensor_value_info('A', TensorProto.FLOAT, [floatcount])
B = make_tensor_value_info('B', TensorProto.FLOAT, [floatcount])
 
out = make_tensor_value_info('out', TensorProto.FLOAT, [floatcount]) 


#-------------------------------------------------------
m1 = make_node('Mul',
                inputs=['A','B'],
                outputs= ['AB'])

s1 = make_node('GatherElements', 
                  inputs=['AB', 'selector'],
                  outputs= ['sAB'],
                  axis=0 )

realparts = make_node('Sub',
                  inputs=['AB','sAB'],
                  outputs= ['reals'])

#------------------------------------------------------- 
s2 = make_node('GatherElements', 
                  inputs=['B', 'selector'],
                  outputs= ['sB'],
                  axis=0 )
m2 = make_node('Mul',
                inputs=['A','sB'],
                outputs= ['sBA'])

s3 = make_node('GatherElements', 
                  inputs=['sBA', 'selector'],
                  outputs= ['ssBA'],
                  axis=0 )

imagparts = make_node('Add',
                  inputs=['sBA','ssBA'],
                  outputs= ['imags'])

#-------------------------------------------------------

summ = make_node('Where',
                 inputs=['selreal','reals','imags'],
                 outputs=['out'])

#-------------------------------------------------------
graph = make_graph(nodes=[m1,s1,realparts,s2,m2,s3,imagparts,summ], 
                   name='ComplexMult', 
                   inputs=[A,B], 
                   outputs=[out],
                   initializer=[selector , selreal])
onnx_model = make_model(graph)
check_model(onnx_model)
# print(onnx_model)

converted_model = version_converter.convert_version(onnx_model, 21)
# The serialization
with open("cpxmult.onnx", "wb") as f:
    f.write(converted_model.SerializeToString())


#sess = ReferenceEvaluator(onnx_model)
#vA =numpy.array([1,2,3,4], dtype=numpy.float32)
#vB =numpy.array([0,1,0,1], dtype=numpy.float32)


#feeds = {'A': vA, 'B': vB }
#print(sess.run(None, feeds))
