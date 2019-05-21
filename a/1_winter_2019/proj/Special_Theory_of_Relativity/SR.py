def calculate_speed(x1,x2):
    c = 3 * 10e8
    v1 = x1/3.6/c
    v2 = x2/3.6/c
    v=(v1 + v2)/(1+(v1*v2/(c**2)))
    return v*c*3.6

a1= 96
a2= 96
a3=calculate_speed(a1,a2)
print(a3)

a4=calculate_speed(a1*10,a2*10)


