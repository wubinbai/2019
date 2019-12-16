from string import Template
t = Template("Hey, $name!")
print(t.substitute(name='wubin'))
print('When to use t emplate string? When yoiu need security. E.g. user input')
