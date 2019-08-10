def lcm(x, y):
    if x > y:
        greater = x
    else:
        greater = y
    while(True):
        if((greater % x < 0.001) and (greater % y <0.001)):
            lcm = greater
            break
        greater += 0.01
    return lcm  
  
  
num1 = float(input("Enter first number: "))  
num2 = float(input("Enter second number: "))  
print("The L.C.M. of", num1,"and", num2,"is", lcm(num1, num2))

