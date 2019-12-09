import math

class Pizza:
    def __init__(self, radius, ingredients):
        self.ingredients = ingredients
        self.radius = radius

    def __repr__(self):
        return (f'Pizza({self.radius!r}, '
                f'Pizza({self.ingredients!r})')

    def area(self):
        return self.circle_area(self.radius)

    @staticmethod
    def circle_area(r):
        return r ** 2 * math.pi
    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella','tomatoes','ham'])
    
    def

pizza = Pizza(['cheese','tomato'])
print(pizza)
print(Pizza.margherita())
print(Pizza.prosciutto())

