"""Implement backward pass for the above two networks. Print the gradient values for each
neuron in each layer."""

import operator
import math
import sympy as sp


#
# x = -2
# y = 5
# z = -4

binary_ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

unary_ops = {
"e": math.exp,      # e^x
    "log": math.log,     # natural log, ln(x)
    "sqrt": math.sqrt,   # square root
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tanh": math.tanh,
    "neg": lambda x: -x,      # negation, -x
    "sigmoid": sigmoid,
}
#
#
# def normal_calc():
#     z = int(input("enter the value for z"))
#     op_1 = input("enter op 1")
#     y = int(input("enter the value for y"))
#     first_op = ops[op_1](y,z)
#     reg_exp = [y,op_1,z]
#
#     op2 = input("enter op 2")
#     x = int(input("enter the value for x"))
#     second_op = ops[op2](x,first_op)
#     reg_exp_2 = [x,op2,first_op]
#     return reg_exp,reg_exp_2,first_op,second_op
#
# # df = normal_calc()
# # print(df)
#
# def derv_func(reg_exp,reg_exp_2,first_op,second_op):
#     integers =[]
#     for i in (reg_exp):
#
#         if i == int:
#             config={
#                 'y': i,
#
#             }
#
#
#     return
#
# reg_exp,reg_exp_2,first_op,second_op = normal_calc()
# print(derv_func(reg_exp,reg_exp_2,first_op,second_op))

def op_grad_binary(op,left,right):
    #Left and right are 2 nodes joined together by an operation
    if op == "+":
        return 1,1
    elif op == "-":
        return 1,-1
    elif op == "*":
        return right,left
    elif op == "/":
        return 1/right,-left/(right**2)
    elif op == "^":
        return right * (left ** (right - 1)),None


def op_grad_unary(op, left, result):
    if op == "e":
        # d(e^x)/dx = e^x we already computed
        return result
    elif op == "log":
        return 1 / left
    elif op == "sqrt":
        return 1 / (2 * result)         # result is already sqrt(left)
    elif op == "sin":
        return math.cos(left)
    elif op == "cos":
        return -math.sin(left)
    elif op == "tan":
        return 1 / (math.cos(left) ** 2)
    elif op == "tanh":
        return 1 - result ** 2           # result is already tanh(left)
    elif op == "neg":
        return -1
    elif op == "sigmoid":
        return result * (1 - result)

#Forward pass
result = float(input("Enter the first value:"))
number_of_ops = int(input("Enter the number of operations:"))

trace = []

for i in range(number_of_ops):
    operation = input(f"Enter the operator {i+1} (+ - * / ^ or e for exponential) : ")

    if operation in unary_ops:
        #for unary second value is not required
        new_result = unary_ops[operation](result)
        trace.append({"type": "unary", "left": result, "op": operation, "result": new_result})
    else:
        value_of_int = float(input(f"Enter the value {i+1}: "))
        new_result = binary_ops[operation](result,value_of_int)
        trace.append({"type": "binary", "left": result, "op": operation, "right": value_of_int, "result": new_result})
    result = new_result

print("\n Final result:",result)


#Backward Pass

grad = 1
print("\n____Gradients_____")
for step in reversed(trace):
#     d_left, d_right = op_grad(step["op"],step["left"],step["right"])
#     grad_right = grad * d_right if d_right is not None else None   # This is calculating the local derivative , how much does the final output change when the right input is changed
#     grad_left = grad * d_left  # This is calculating the local derivative , how much does the final output change when the left input is changed
#
#     print(f"Step ({step['left']} {step['op']} {step['right']}): "
#           f"d(final)/d(left) = {grad_left} , d(final)/d(right) = {grad_right}")
#
#     grad = grad_left
#
# print(f"\nd(final)/d(first value) = {grad}")
    if step["type"] == "unary":
         d_left = op_grad_unary(step["op"], step["left"], step["result"])
         grad_left = grad * d_left
         print(f"Step ({step['op']}({step['left']})): d(final)/d(left) = {grad_left}")

    else:
        d_left, d_right = op_grad_binary(step["op"], step["left"], step["right"])
        grad_right = grad * d_right if d_right is not None else None
        grad_left = grad * d_left
        print(f"Step ({step['left']} {step['op']} {step['right']}): "
          f"d(final)/d(left) = {grad_left}, d(final)/d(right) = {grad_right}")

    grad = grad_left

print(f"\nd(final)/d(first value) = {grad}")



