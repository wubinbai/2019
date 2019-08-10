

Some more options. You can decided which is the "easiest way".

Remove the first 2 characters of every line:

:%normal 2x

Remove first 2 characters of every line, only if they're spaces:

:%s/^  /

Note that the last slash is optional, and is only here so that you can see the two spaces. Without the slash, it's only 7 characters, including the :.

Move indentation to left for every line:

:%normal <<


