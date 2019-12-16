def greet(name, question):
    return ("hello, " + name + "! How's it " + question + "?")

import dis
print(dis.dis(greet))
