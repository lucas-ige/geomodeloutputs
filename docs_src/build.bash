#!/bin/bash
#
# Build the documentation.

\make html
\rsync -vrptlh --exclude=.DS_Store --exclude=.git --delete --update ./bld/html/ ../docs
\touch ../docs/.nojekyll
