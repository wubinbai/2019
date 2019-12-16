class Test:
    def __init__(self):
        self.foo = 'this is foo'
        self._bar = 'this is _bar'
        self.__baz = 'this is __baz'


class ExtendedTest(Test):
    def __init__(self):
        super().__init__()
        self.foo = 'overridden'
        self._bar = 'overdridden'
        self.__baz = 'overridden'

class ManglingTest:
    def __init__(self):
        self.__mangled = 'hello'
    def get_mangled(self):
        return self.__mangled

class MangledMethod:
    def __method(self):
        return 42

    def call_it(self):
        return self.__method()

_MangledGlobal__mangled = 23
class MangledGlobal:
    def test(self):
        return __mangled

