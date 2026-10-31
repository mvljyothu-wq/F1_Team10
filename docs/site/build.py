"""Builds the presentation page from page.src.html + results data.

  artifact.html  - content for the hosted page (the host adds <html>/<head>)
  index.html     - standalone file that opens directly in a browser, offline too
"""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "page.src.html"), encoding="utf-8").read()
data = open(os.path.join(here, "data.json"), encoding="utf-8").read()
page = src.replace("__DATA__", data)
open(os.path.join(here, "artifact.html"), "w", encoding="utf-8").write(page)
standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
              + page.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
open(os.path.join(here, "index.html"), "w", encoding="utf-8").write(standalone)
print("built", len(page), "bytes")
