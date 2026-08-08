"""Implement backward pass for the above two networks. Print the gradient values for each
neuron in each layer."""

import operator

import sympy as sp


#
# x = -2
# y = 5
# z = -4

ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.floordiv,
    "^": operator.mod,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}


def normal_calc():
    z = int(input("enter the value for z"))
    op_1 = input("enter op 1")
    y = int(input("enter the value for y"))
    first_op = ops[op_1](y,z)
    reg_exp = [z,op_1,y]

    op2 = input("enter op 2")
    x = int(input("enter the value for x"))
    second_op = ops[op2](x,first_op)
    reg_exp_2 = [x,op2,first_op]
    return reg_exp,reg_exp_2,first_op,second_op

# df = normal_calc()
# print(df)

def derv_func(reg_exp,reg_exp_2,first_op,second_op):
    
