??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8??
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	?y*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	?y
l

dense/biasVarHandleOp*
shared_name
dense/bias*
dtype0*
_output_shapes
: *
shape:y
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:y
y
dense_1/kernelVarHandleOp*
shape:	y?*
shared_namedense_1/kernel*
dtype0*
_output_shapes
: 
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	y?
q
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
dtype0*
_output_shapes
: *
shape:?
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:?
n
Adadelta/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
dtype0	*
_output_shapes
: 
p
Adadelta/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
dtype0*
_output_shapes
: 
?
Adadelta/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
dtype0*
_output_shapes
: 
l
Adadelta/rhoVarHandleOp*
shape: *
shared_nameAdadelta/rho*
dtype0*
_output_shapes
: 
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
dtype0*
_output_shapes
: 
?
 Adadelta/dense/kernel/accum_gradVarHandleOp*1
shared_name" Adadelta/dense/kernel/accum_grad*
dtype0*
_output_shapes
: *
shape:	?y
?
4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense/kernel/accum_grad*
dtype0*
_output_shapes
:	?y
?
Adadelta/dense/bias/accum_gradVarHandleOp*
dtype0*
_output_shapes
: *
shape:y*/
shared_name Adadelta/dense/bias/accum_grad
?
2Adadelta/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_grad*
dtype0*
_output_shapes
:y
?
"Adadelta/dense_1/kernel/accum_gradVarHandleOp*
dtype0*
_output_shapes
: *
shape:	y?*3
shared_name$"Adadelta/dense_1/kernel/accum_grad
?
6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1/kernel/accum_grad*
dtype0*
_output_shapes
:	y?
?
 Adadelta/dense_1/bias/accum_gradVarHandleOp*
shape:?*1
shared_name" Adadelta/dense_1/bias/accum_grad*
dtype0*
_output_shapes
: 
?
4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_1/bias/accum_grad*
dtype0*
_output_shapes	
:?
?
Adadelta/dense/kernel/accum_varVarHandleOp*
dtype0*
_output_shapes
: *
shape:	?y*0
shared_name!Adadelta/dense/kernel/accum_var
?
3Adadelta/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/kernel/accum_var*
dtype0*
_output_shapes
:	?y
?
Adadelta/dense/bias/accum_varVarHandleOp*
shape:y*.
shared_nameAdadelta/dense/bias/accum_var*
dtype0*
_output_shapes
: 
?
1Adadelta/dense/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense/bias/accum_var*
dtype0*
_output_shapes
:y
?
!Adadelta/dense_1/kernel/accum_varVarHandleOp*
shape:	y?*2
shared_name#!Adadelta/dense_1/kernel/accum_var*
dtype0*
_output_shapes
: 
?
5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1/kernel/accum_var*
dtype0*
_output_shapes
:	y?
?
Adadelta/dense_1/bias/accum_varVarHandleOp*
shape:?*0
shared_name!Adadelta/dense_1/bias/accum_var*
dtype0*
_output_shapes
: 
?
3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_1/bias/accum_var*
dtype0*
_output_shapes	
:?

NoOpNoOp
?
ConstConst"/device:CPU:0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
	keras_api
trainable_variables

signatures
	regularization_losses
R

	variables
	keras_api
trainable_variables
regularization_losses
h

kernel
bias
	variables
	keras_api
trainable_variables
regularization_losses
h

kernel
bias
	variables
	keras_api
trainable_variables
regularization_losses
?
iter
	decay
learning_rate
rho
accum_grad.
accum_grad/
accum_grad0
accum_grad1	accum_var2	accum_var3	accum_var4	accum_var5

0
1
2
3
?

layers
layer_regularization_losses
	variables
	regularization_losses
 non_trainable_variables
trainable_variables
!metrics

0
1
2
3
 
 
 
?

"layers
#layer_regularization_losses

	variables
regularization_losses
$non_trainable_variables
trainable_variables
%metrics
 
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
?

&layers
'layer_regularization_losses
	variables
regularization_losses
(non_trainable_variables
trainable_variables
)metrics

0
1
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
?

*layers
+layer_regularization_losses
	variables
regularization_losses
,non_trainable_variables
trainable_variables
-metrics

0
1
 
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUE Adadelta/dense/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_1/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_1/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_1/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/dense_1/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
|
serving_default_input_1Placeholder*
shape:??????????*
dtype0*(
_output_shapes
:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/bias**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin	
2*,
_gradient_op_typePartitionedCall-63580*,
f'R%
#__inference_signature_wrapper_63397*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOp4Adadelta/dense/kernel/accum_grad/Read/ReadVariableOp2Adadelta/dense/bias/accum_grad/Read/ReadVariableOp6Adadelta/dense_1/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_1/bias/accum_grad/Read/ReadVariableOp3Adadelta/dense/kernel/accum_var/Read/ReadVariableOp1Adadelta/dense/bias/accum_var/Read/ReadVariableOp5Adadelta/dense_1/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_1/bias/accum_var/Read/ReadVariableOpConst**
config_proto

GPU 

CPU2J 8*
Tin
2	*
_output_shapes
: *,
_gradient_op_typePartitionedCall-63618*'
f"R 
__inference__traced_save_63617*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rho Adadelta/dense/kernel/accum_gradAdadelta/dense/bias/accum_grad"Adadelta/dense_1/kernel/accum_grad Adadelta/dense_1/bias/accum_gradAdadelta/dense/kernel/accum_varAdadelta/dense/bias/accum_var!Adadelta/dense_1/kernel/accum_varAdadelta/dense_1/bias/accum_var**
f%R#
!__inference__traced_restore_63678*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-63679??
?
?
'__inference_model_1_layer_call_fn_63410

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-63332*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_63331*
Tout
2**
config_proto

GPU 

CPU2J 8*,
_output_shapes
:??????????: : *
Tin	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_63502

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?yi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:yv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????y?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
'__inference_model_1_layer_call_fn_63383
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*,
_output_shapes
:??????????: : *,
_gradient_op_typePartitionedCall-63374*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_63373*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
?D
?	
!__inference__traced_restore_63678
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias$
 assignvariableop_4_adadelta_iter%
!assignvariableop_5_adadelta_decay-
)assignvariableop_6_adadelta_learning_rate#
assignvariableop_7_adadelta_rho7
3assignvariableop_8_adadelta_dense_kernel_accum_grad5
1assignvariableop_9_adadelta_dense_bias_accum_grad:
6assignvariableop_10_adadelta_dense_1_kernel_accum_grad8
4assignvariableop_11_adadelta_dense_1_bias_accum_grad7
3assignvariableop_12_adadelta_dense_kernel_accum_var5
1assignvariableop_13_adadelta_dense_bias_accum_var9
5assignvariableop_14_adadelta_dense_1_kernel_accum_var7
3assignvariableop_15_adadelta_dense_1_bias_accum_var
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?	
RestoreV2/tensor_namesConst"/device:CPU:0*?	
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_adadelta_iterIdentity_4:output:0*
dtype0	*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_adadelta_decayIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_adadelta_learning_rateIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adadelta_rhoIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp3assignvariableop_8_adadelta_dense_kernel_accum_gradIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp1assignvariableop_9_adadelta_dense_bias_accum_gradIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_adadelta_dense_1_kernel_accum_gradIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adadelta_dense_1_bias_accum_gradIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp3assignvariableop_12_adadelta_dense_kernel_accum_varIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adadelta_dense_bias_accum_varIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0?
AssignVariableOp_14AssignVariableOp5assignvariableop_14_adadelta_dense_1_kernel_accum_varIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp3assignvariableop_15_adadelta_dense_1_bias_accum_varIdentity_15:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2: : : : : : : :	 :
 : : : : : : :+ '
%
_user_specified_namefile_prefix: 
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_63373

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity

identity_1

identity_2??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????y*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63193*5
f0R.
,__inference_dense_activity_regularizer_63158*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tin
2u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-63235*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:???????????
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63249*7
f2R0
.__inference_dense_1_activity_regularizer_63166*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tin
2y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: {
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*
_output_shapes
: *

SrcT0?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
_output_shapes
: *
T0?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:???????????

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
: *
T0?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
: *
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
?
?
'__inference_dense_1_layer_call_fn_63525

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:??????????*,
_gradient_op_typePartitionedCall-63235*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????y::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_63234

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	y?j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????y::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_63183

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?yi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:yv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????y?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
F
,__inference_dense_activity_regularizer_63158
self
identityJ
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: E
IdentityIdentityConst:output:0*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
::$  

_user_specified_nameself
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_63300
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity

identity_1

identity_2??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????y*
Tin
2?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-63193*5
f0R.
,__inference_dense_activity_regularizer_63158*
Tout
2u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*
_output_shapes
: *

SrcT0?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
_output_shapes
: *
T0?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin
2*,
_gradient_op_typePartitionedCall-63235*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63249*7
f2R0
.__inference_dense_1_activity_regularizer_63166*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
_output_shapes
:*
T0y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*(
_output_shapes
:??????????*
T0?

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: ?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: 
?,
?
__inference__traced_save_63617
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop?
;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_dense_bias_accum_grad_read_readvariableopA
=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop>
:savev2_adadelta_dense_kernel_accum_var_read_readvariableop<
8savev2_adadelta_dense_bias_accum_var_read_readvariableop@
<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_cfa79d18c07642d3a4f64d1510630090/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?	
SaveV2/tensor_namesConst"/device:CPU:0*?	
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop;savev2_adadelta_dense_kernel_accum_grad_read_readvariableop9savev2_adadelta_dense_bias_accum_grad_read_readvariableop=savev2_adadelta_dense_1_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_1_bias_accum_grad_read_readvariableop:savev2_adadelta_dense_kernel_accum_var_read_readvariableop8savev2_adadelta_dense_bias_accum_var_read_readvariableop<savev2_adadelta_dense_1_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_1_bias_accum_var_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapesw
u: :	?y:y:	y?:?: : : : :	?y:y:	y?:?:	?y:y:	y?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : 
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_63270
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity

identity_1

identity_2??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????y*
Tin
2*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183*
Tout
2?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63193*5
f0R.
,__inference_dense_activity_regularizer_63158*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*
_output_shapes
: *

SrcT0?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin
2*,
_gradient_op_typePartitionedCall-63235?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*7
f2R0
.__inference_dense_1_activity_regularizer_63166*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-63249y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*
_output_shapes
: *

SrcT0?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:???????????

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: ?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: 
?
?
#__inference_signature_wrapper_63397
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-63390*)
f$R"
 __inference__wrapped_model_63150*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*(
_output_shapes
:???????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : : 
?
H
.__inference_dense_1_activity_regularizer_63166
self
identityJ
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
::$  

_user_specified_nameself
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_63331

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity

identity_1

identity_2??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????y?
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63193*5
f0R.
,__inference_dense_activity_regularizer_63158*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin
2*,
_gradient_op_typePartitionedCall-63235*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2?
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*7
f2R0
.__inference_dense_1_activity_regularizer_63166*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-63249y
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:???????????

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: ?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
: *
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?*
?
 __inference__wrapped_model_63150
input_10
,model_1_dense_matmul_readvariableop_resource1
-model_1_dense_biasadd_readvariableop_resource2
.model_1_dense_1_matmul_readvariableop_resource3
/model_1_dense_1_biasadd_readvariableop_resource
identity??$model_1/dense/BiasAdd/ReadVariableOp?#model_1/dense/MatMul/ReadVariableOp?&model_1/dense_1/BiasAdd/ReadVariableOp?%model_1/dense_1/MatMul/ReadVariableOp?
#model_1/dense/MatMul/ReadVariableOpReadVariableOp,model_1_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?y?
model_1/dense/MatMulMatMulinput_1+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp-model_1_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:y?
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yl
model_1/dense/ReluRelumodel_1/dense/BiasAdd:output:0*'
_output_shapes
:?????????y*
T0l
'model_1/dense/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: w
'model_1/dense/ActivityRegularizer/ShapeShape model_1/dense/Relu:activations:0*
T0*
_output_shapes
:
5model_1/dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:?
7model_1/dense/ActivityRegularizer/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:?
7model_1/dense/ActivityRegularizer/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
/model_1/dense/ActivityRegularizer/strided_sliceStridedSlice0model_1/dense/ActivityRegularizer/Shape:output:0>model_1/dense/ActivityRegularizer/strided_slice/stack:output:0@model_1/dense/ActivityRegularizer/strided_slice/stack_1:output:0@model_1/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask?
&model_1/dense/ActivityRegularizer/CastCast8model_1/dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
)model_1/dense/ActivityRegularizer/truedivRealDiv0model_1/dense/ActivityRegularizer/Const:output:0*model_1/dense/ActivityRegularizer/Cast:y:0*
_output_shapes
: *
T0?
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	y??
model_1/dense_1/MatMulMatMul model_1/dense/Relu:activations:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????n
)model_1/dense_1/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: y
)model_1/dense_1/ActivityRegularizer/ShapeShape model_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
7model_1/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:?
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:?
9model_1/dense_1/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
1model_1/dense_1/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_1/ActivityRegularizer/Shape:output:0@model_1/dense_1/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_1/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ?
(model_1/dense_1/ActivityRegularizer/CastCast:model_1/dense_1/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
+model_1/dense_1/ActivityRegularizer/truedivRealDiv2model_1/dense_1/ActivityRegularizer/Const:output:0,model_1/dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity model_1/dense_1/BiasAdd:output:0%^model_1/dense/BiasAdd/ReadVariableOp$^model_1/dense/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#model_1/dense/MatMul/ReadVariableOp#model_1/dense/MatMul/ReadVariableOp2L
$model_1/dense/BiasAdd/ReadVariableOp$model_1/dense/BiasAdd/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp: :' #
!
_user_specified_name	input_1: : : 
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_63535

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	y?j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????y::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
%__inference_dense_layer_call_fn_63518

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????y*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
'__inference_model_1_layer_call_fn_63421

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-63374*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_63373*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*,
_output_shapes
:??????????: : ?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
?
?
'__inference_model_1_layer_call_fn_63341
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-63332*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_63331*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*,
_output_shapes
:??????????: : ?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
?

?
F__inference_dense_1_layer_call_and_return_all_conditional_losses_63544

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-63235*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_63234*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tin
2?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63249*7
f2R0
.__inference_dense_1_activity_regularizer_63166*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: ?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????k

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????y::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?*
?
B__inference_model_1_layer_call_and_return_conditional_losses_63491

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?yu
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:y?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????yd
dense/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
!dense/ActivityRegularizer/truedivRealDiv(dense/ActivityRegularizer/Const:output:0"dense/ActivityRegularizer/Cast:y:0*
_output_shapes
: *
T0?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	y??
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
!dense_1/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: i
!dense_1/ActivityRegularizer/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: ?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*
_output_shapes
: *

SrcT0?
#dense_1/ActivityRegularizer/truedivRealDiv*dense_1/ActivityRegularizer/Const:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentitydense_1/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0?

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
?*
?
B__inference_model_1_layer_call_and_return_conditional_losses_63456

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?yu
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:y?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????yd
dense/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ?
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
!dense/ActivityRegularizer/truedivRealDiv(dense/ActivityRegularizer/Const:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	y??
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
!dense_1/ActivityRegularizer/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: i
!dense_1/ActivityRegularizer/ShapeShapedense_1/BiasAdd:output:0*
_output_shapes
:*
T0y
/dense_1/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:{
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0?
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: ?
#dense_1/ActivityRegularizer/truedivRealDiv*dense_1/ActivityRegularizer/Const:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentitydense_1/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:???????????

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:??????????::::2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
?

?
D__inference_dense_layer_call_and_return_all_conditional_losses_63511

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-63189*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_63183*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????y?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-63193*5
f0R.
,__inference_dense_activity_regularizer_63158*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: ?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????yk

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????<
dense_11
StatefulPartitionedCall:0??????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:?x
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
	keras_api
trainable_variables

signatures
	regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
8_default_save_signature"?
_tf_keras_model?{"class_name": "Model", "training_config": {"loss": "mean_squared_error", "metrics": [], "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"rho": 0.949999988079071, "epsilon": 1e-07, "decay": 0.0, "learning_rate": 0.0010000000474974513, "name": "Adadelta"}}, "weighted_metrics": null}, "config": {"output_layers": [["dense_1", 0, 0]], "layers": [{"class_name": "InputLayer", "inbound_nodes": [], "config": {"sparse": false, "dtype": "float32", "batch_input_shape": [null, 840], "name": "input_1"}, "name": "input_1"}, {"class_name": "Dense", "inbound_nodes": [[["input_1", 0, 0, {}]]], "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "activation": "relu", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "use_bias": true, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 121}, "name": "dense"}, {"class_name": "Dense", "inbound_nodes": [[["dense", 0, 0, {}]]], "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "use_bias": true, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_1", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 840}, "name": "dense_1"}], "input_layers": [["input_1", 0, 0]], "name": "model_1"}, "backend": "tensorflow", "name": "model_1", "expects_training_arg": true, "trainable": true, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "model_config": {"class_name": "Model", "config": {"output_layers": [["dense_1", 0, 0]], "layers": [{"class_name": "InputLayer", "inbound_nodes": [], "config": {"sparse": false, "dtype": "float32", "batch_input_shape": [null, 840], "name": "input_1"}, "name": "input_1"}, {"class_name": "Dense", "inbound_nodes": [[["input_1", 0, 0, {}]]], "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "activation": "relu", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "use_bias": true, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 121}, "name": "dense"}, {"class_name": "Dense", "inbound_nodes": [[["dense", 0, 0, {}]]], "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "use_bias": true, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_1", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 840}, "name": "dense_1"}], "input_layers": [["input_1", 0, 0]], "name": "model_1"}}}
?

	variables
	keras_api
trainable_variables
regularization_losses
9__call__
*:&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "config": {"sparse": false, "dtype": "float32", "batch_input_shape": [null, 840], "name": "input_1"}, "name": "input_1", "expects_training_arg": true, "trainable": true, "dtype": "float32", "batch_input_shape": [null, 840]}
?

kernel
bias
	variables
	keras_api
trainable_variables
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "use_bias": true, "bias_constraint": null, "activation": "relu", "kernel_constraint": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 121}, "input_spec": {"class_name": "InputSpec", "config": {"axes": {"-1": 840}, "shape": null, "ndim": null, "max_ndim": null, "dtype": null, "min_ndim": 2}}, "name": "dense", "expects_training_arg": false, "trainable": true, "dtype": "float32", "batch_input_shape": null}
?

kernel
bias
	variables
	keras_api
trainable_variables
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "config": {"bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "use_bias": true, "bias_constraint": null, "activation": "linear", "kernel_constraint": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l2": 0.0, "l1": 0.0}}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense_1", "trainable": true, "dtype": "float32", "kernel_regularizer": null, "units": 840}, "input_spec": {"class_name": "InputSpec", "config": {"axes": {"-1": 121}, "shape": null, "ndim": null, "max_ndim": null, "dtype": null, "min_ndim": 2}}, "name": "dense_1", "expects_training_arg": false, "trainable": true, "dtype": "float32", "batch_input_shape": null}
?
iter
	decay
learning_rate
rho
accum_grad.
accum_grad/
accum_grad0
accum_grad1	accum_var2	accum_var3	accum_var4	accum_var5"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
?

layers
layer_regularization_losses
	variables
	regularization_losses
 non_trainable_variables
trainable_variables
!metrics
6__call__
&7"call_and_return_conditional_losses
*7&call_and_return_all_conditional_losses
8_default_save_signature"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
,
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

"layers
#layer_regularization_losses

	variables
regularization_losses
$non_trainable_variables
trainable_variables
%metrics
9__call__
&:"call_and_return_conditional_losses
*:&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	?y2dense/kernel
:y2
dense/bias
.
0
1"
trackable_list_wrapper
?

&layers
'layer_regularization_losses
	variables
regularization_losses
(non_trainable_variables
trainable_variables
)metrics
&@"call_and_return_conditional_losses
;__call__
Aactivity_regularizer_fn
*<&call_and_return_all_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
!:	y?2dense_1/kernel
:?2dense_1/bias
.
0
1"
trackable_list_wrapper
?

*layers
+layer_regularization_losses
	variables
regularization_losses
,non_trainable_variables
trainable_variables
-metrics
&B"call_and_return_conditional_losses
=__call__
Cactivity_regularizer_fn
*>&call_and_return_all_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
1:/	?y2 Adadelta/dense/kernel/accum_grad
*:(y2Adadelta/dense/bias/accum_grad
3:1	y?2"Adadelta/dense_1/kernel/accum_grad
-:+?2 Adadelta/dense_1/bias/accum_grad
0:.	?y2Adadelta/dense/kernel/accum_var
):'y2Adadelta/dense/bias/accum_var
2:0	y?2!Adadelta/dense_1/kernel/accum_var
,:*?2Adadelta/dense_1/bias/accum_var
?2?
'__inference_model_1_layer_call_fn_63341
'__inference_model_1_layer_call_fn_63410
'__inference_model_1_layer_call_fn_63383
'__inference_model_1_layer_call_fn_63421?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_63270
B__inference_model_1_layer_call_and_return_conditional_losses_63491
B__inference_model_1_layer_call_and_return_conditional_losses_63300
B__inference_model_1_layer_call_and_return_conditional_losses_63456?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_63150?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_63518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_layer_call_and_return_all_conditional_losses_63511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_63525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_1_layer_call_and_return_all_conditional_losses_63544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2B0
#__inference_signature_wrapper_63397input_1
?2?
@__inference_dense_layer_call_and_return_conditional_losses_63502?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_activity_regularizer_63158?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_63535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_dense_1_activity_regularizer_63166?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?{
'__inference_dense_1_layer_call_fn_63525P/?,
%?"
 ?
inputs?????????y
? "????????????
'__inference_model_1_layer_call_fn_63341\9?6
/?,
"?
input_1??????????
p

 
? "????????????
B__inference_model_1_layer_call_and_return_conditional_losses_63456?8?5
.?+
!?
inputs??????????
p

 
? "B??
?
0??????????
?
?	
1/0 
?	
1/1 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_63535]/?,
%?"
 ?
inputs?????????y
? "&?#
?
0??????????
? y
%__inference_dense_layer_call_fn_63518P0?-
&?#
!?
inputs??????????
? "??????????yY
,__inference_dense_activity_regularizer_63158)?
?
?
self
? "? ?
B__inference_model_1_layer_call_and_return_conditional_losses_63300?9?6
/?,
"?
input_1??????????
p 

 
? "B??
?
0??????????
?
?	
1/0 
?	
1/1 ?
'__inference_model_1_layer_call_fn_63410[8?5
.?+
!?
inputs??????????
p

 
? "????????????
#__inference_signature_wrapper_63397x<?9
? 
2?/
-
input_1"?
input_1??????????"2?/
-
dense_1"?
dense_1??????????[
.__inference_dense_1_activity_regularizer_63166)?
?
?
self
? "? ?
F__inference_dense_1_layer_call_and_return_all_conditional_losses_63544k/?,
%?"
 ?
inputs?????????y
? "4?1
?
0??????????
?
?	
1/0 ?
@__inference_dense_layer_call_and_return_conditional_losses_63502]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????y
? ?
'__inference_model_1_layer_call_fn_63421[8?5
.?+
!?
inputs??????????
p 

 
? "????????????
B__inference_model_1_layer_call_and_return_conditional_losses_63491?8?5
.?+
!?
inputs??????????
p 

 
? "B??
?
0??????????
?
?	
1/0 
?	
1/1 ?
B__inference_model_1_layer_call_and_return_conditional_losses_63270?9?6
/?,
"?
input_1??????????
p

 
? "B??
?
0??????????
?
?	
1/0 
?	
1/1 ?
 __inference__wrapped_model_63150m1?.
'?$
"?
input_1??????????
? "2?/
-
dense_1"?
dense_1???????????
'__inference_model_1_layer_call_fn_63383\9?6
/?,
"?
input_1??????????
p 

 
? "????????????
D__inference_dense_layer_call_and_return_all_conditional_losses_63511k0?-
&?#
!?
inputs??????????
? "3?0
?
0?????????y
?
?	
1/0 