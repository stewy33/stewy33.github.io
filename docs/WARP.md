# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a personal/professional website hosted at https://www.stewyslocum.com using GitHub Pages with Jekyll. The site includes a homepage, blog posts with mathematical content, and tag-based organization.

## Development Commands

### Local Development
```bash
bundle exec jekyll serve
```
This builds the site and serves it locally with live reload at http://localhost:4000.

### Production Build
```bash
bundle exec jekyll build
```
Then manually copy the contents of `_site/` to `docs/` for GitHub Pages deployment.

## Architecture

### Jekyll Structure
- **`_layouts/`**: HTML templates
  - `default.html` - Base template with header/footer
  - `post.html` - Blog post template with navigation, comments support, and tags
  - `tagpage.html` - Template for tag archive pages
- **`_includes/`**: Reusable HTML components (header, footer, head, tags, disqus)
- **`_posts/`**: Blog posts in markdown with YAML frontmatter
  - Post filenames follow `YYYY-MM-DD-title.md` convention
  - Posts support tags, comments (via Disqus), and LaTeX math rendering
- **`_sass/`**: SCSS stylesheets
  - Uses Minima theme as base
  - Custom fonts and rouge-github syntax highlighting
- **`_bibliography/`**: Bibliography data for jekyll-scholar citations
- **`assets/`**: Static files (css, images, fonts, data like resume PDF)
- **`tag/`**: Tag archive pages
- **`wip/`**: Work-in-progress posts not yet published
- **`docs/`**: Production build output for GitHub Pages

### Key Plugins
- `jekyll-feed`: RSS feed generation
- `jekyll-scholar`: Academic citation support
- `rouge`: Syntax highlighting
- `kramdown`: Markdown processor with GFM support and math rendering

### Content Notes
- Blog posts use extensive LaTeX math notation (requires kramdown with MathJax/KaTeX)
- Posts include academic citations via jekyll-scholar
- Tag system allows filtering posts by topic (e.g., optimization, review, long-read)
