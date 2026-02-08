rm -r docs site_
bundle exec jekyll build
cp -r _site docs
rm docs/README.md docs/WARP.md