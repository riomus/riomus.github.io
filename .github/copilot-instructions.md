# Copilot Instructions for riomus.github.io

This repository is a Jekyll site using the Hydeout theme, customized for personal or project pages. Follow these guidelines to be productive as an AI coding agent:

## Architecture Overview
- **Jekyll-based static site**: Content is organized in Markdown files (`_posts/`, root-level `.md` files) and rendered via layouts in `_layouts/`.
- **Hydeout theme**: The site uses the Hydeout theme, with customizations possible via SASS variables and partials in `_includes/` and `_sass/`.
- **Assets**: Custom styles go in `assets/css/main.scss`. Images are in `assets/img/`.
- **Site configuration**: `_config.yml` controls site-wide settings, including theme, pagination, Disqus, and Google Analytics.

## Key Workflows
- **Local development**: Use `bundle install` to install dependencies, then `bundle exec jekyll serve` to run the site locally.
- **Theme customization**: Override SASS variables in `assets/css/main.scss` and import Hydeout's SCSS. See `_sass/hydeout/_variables.scss` for available variables.
- **Content creation**:
  - Posts: Add Markdown files to `_posts/` with appropriate front matter.
  - Pages: Add Markdown or HTML files at the root, using layouts from `_layouts/`.
  - Tags/Categories: Create pages with `tags` or `category` layout for sidebar navigation.
- **Sidebar navigation**: Pages must have `sidebar_link: true` in front matter to appear in the sidebar. Use `sidebar_sort_order` to control order.
- **Disqus/Analytics**: Enable by adding `disqus.shortname` or `google_analytics` to `_config.yml`.

## Project-Specific Patterns
- **Partial overrides**: Customize site sections by editing or replacing files in `_includes/` (e.g., `custom-head.html`, `custom-foot.html`, `copyright.html`).
- **Minimal JavaScript**: JS is only loaded for Disqus and Google Analytics if configured.
- **Flexbox layout**: CSS uses Flexbox; degrades gracefully if unsupported.
- **Pagination**: Controlled via `paginate` in `_config.yml` and `index.html` layout.

## External Integrations
- **Hydeout theme**: Managed via Gemfile and `_config.yml` (`remote_theme: fongandrew/hydeout`).
- **Disqus**: Add `disqus.shortname` to `_config.yml`.
- **Google Analytics**: Add `google_analytics` property to `_config.yml`.

## Examples
- To add a new tag page:
  ```markdown
  ---
  layout: tags
  title: Tags
  ---
  ```
- To override sidebar background color:
  ```scss
  $sidebar-bg-color: #ac4142;
  @import "hydeout";
  ```
- To add a page to the sidebar:
  ```markdown
  ---
  layout: page
  title: My Page
  sidebar_link: true
  sidebar_sort_order: 10
  ---
  ```

## Key Files & Directories
- `_config.yml`: Site configuration
- `Gemfile`: Ruby dependencies
- `_includes/`: HTML partials for customization
- `_sass/`: SASS partials and variables
- `assets/css/main.scss`: Custom styles
- `_layouts/`: Page and post layouts
- `_posts/`: Blog posts

---

If any section is unclear or missing, please provide feedback for further refinement.