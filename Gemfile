source "https://rubygems.org"
# Hello! This is where you manage which Jekyll version is used to run.
# When you want to use a different version, change it below, save the
# file and run `bundle install`. Run Jekyll with `bundle exec`, like so:
#
#     bundle exec jekyll serve
#
# This will help ensure the proper Jekyll version is running.
# Happy Jekylling!
gem "jekyll", "~> 4"
# This is the default theme for new Jekyll sites. You may change this to anything you like.
# gem "minima"
# To upgrade, run `bundle update github-pages`.
# gem "github-pages", group: :jekyll_plugins
# If you have any plugins, put them here!
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem 'jekyll-octicons'
  gem 'jekyll-remote-theme'
  gem 'jekyll-relative-links'
  gem 'jekyll-seo-tag'
  # gem 'jekyll-toc'
  gem 'jekyll-gist'
  gem 'jekyll-paginate'
  gem 'jekyll-sitemap'
end

gem "kramdown-math-katex"

# Pin native-extension gems to Ruby 3.2 bundled versions.
# On Windows, the C toolchain may not compile newer versions.
# On Linux CI (Ruby 3.2 from setup-ruby), these compile fine but the pin
# ensures consistency. CI uses BUNDLE_BUILD__BIGDECIMAL="--use-system-libraries".
gem "json", "2.6.3"
gem "racc", "1.6.2"
gem "bigdecimal", "3.1.3"

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
# gem "wdm", ">= 0.1.0", :install_if => Gem.win_platform?

gem "faraday-retry"

gem "webrick", "~> 1.7"
