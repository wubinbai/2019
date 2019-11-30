def uppercase(f):
    def wrapper_f():
        orig = f()
        modified = orig.upper()
        return modified
    return wrapper_f
@uppercase
def greet():
    return 'hello'

print(greet())
