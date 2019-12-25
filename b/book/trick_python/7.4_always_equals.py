class AlwaysEquals:
    def __eq__(self, other):
        return True
    def __hasH__(self):
        return id(self)


print(AlwaysEquals()==AlwaysEquals())
print(AlwaysEquals()=='3')
print(AlwaysEquals()==123)
