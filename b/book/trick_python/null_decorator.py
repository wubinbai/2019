def null_decorator(f):
    return f

@null_decorator
def greeting():
    return 'hello'

print(greeting())
