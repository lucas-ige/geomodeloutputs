#!/bin/bash
#
# Build the documentation.

\make html
\rsync -vrptlh --exclude=.DS_Store --exclude=.git --delete --update ./bld/html/ ../docs
\touch ../docs/.nojekyll

# We clean up manually the documentation
sed -i "" -e "s/<span class=\"sig-prename descclassname\"><span class=\"pre\">geomodeloutputs\.xarray_accessors\.<\/span><\/span>/<span class=\"sig-prename descclassname\"><span class=\"pre\"><\/span><\/span>/g" ../docs/accessors_commons.html
sed -i "" -e "s/<span class=\"sig-prename descclassname\"><span class=\"pre\">GenericDatasetAccessor\.<\/span><\/span>/<span class=\"sig-prename descclassname\"><span class=\"pre\">ds\.<\/span><\/span>/g" ../docs/accessors_commons.html
