# How to blog

## Images

{:refdef: style="text-align: center;"}
[![bmi](https://upload.wikimedia.org/wikipedia/commons/e/eb/Nutrition-pyramid.jpg){: width="50%" .shadow}](https://commons.wikimedia.org/wiki/File:Nutrition-pyramid.jpg)
{: refdef}
{:refdef: style="text-align: center;"}
<sub><sup>*Fig.3: Food pyramid. [Spmallare](https://commons.wikimedia.org/wiki/File:Nutrition-pyramid.jpg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0), via Wikimedia Commons*
</sup></sub>
{: refdef}

### Dark mode support

- For image invert color in dark mode

  `<img src="path/to/image.png" data-invert-dark="true" ">`

  In markdown, add `{: data-invert-dark="true"}` after the image. If you want this filter to thumbnail image, add `invert_thumbnail_dark: true` in front matter

- Prevent auto inversion (automatically checks if >70% image is white or transparent)

  `<img src="..." data-no-invert>`

- Light card background in dark mode (for diagrams with text on white bg, GIFs, videos): `{: data-transparent-dark="true"}`

- To manually remove white background, run `python _scripts/white2transparent.py "images\multi-task-methods.jpeg" --webp --max-width 1000 --feather 55 --threshold 255`. The same script will also convert to webp format which has much lower image sizes.

## Front Matter Reference

| Key | Type | Where used | Effect |
|---|---|---|---|
| `featured` | boolean | posts | Pins post to top of section and shows Featured badge. |
| `hide` | boolean | posts | Excludes from homepage and related posts. |
| `hidden` | boolean | posts | Excludes from homepage sections. |
| `link` | URL | posts | Renders title as external link with icon (cards + post header). |
| `toc_min` | number | posts | Lower heading level bound for ToC generation. |
| `toc_max` | number | posts | Upper heading level bound for ToC generation. |
| `permalink` | string | pages/posts | Custom output URL path. |
| `badges` | boolean | notebook layout | Show/hide notebook badge row (`false` hides all). |
| `hide_github_badge` | boolean | notebook layout | Hides GitHub badge. |
| `hide_binder_badge` | boolean | notebook layout | Hides Binder badge. |
| `hide_colab_badge` | boolean | notebook layout | Hides Colab badge. |
| `hide_kaggle_badge` | boolean | notebook layout | Hides Kaggle badge. |
| `branch` | string | notebook badges | Branch used in notebook badge URLs (default `master`). |
| `nb_path` | string | notebook badges | Notebook path used by GitHub/Colab/Binder/Kaggle badge links. |

### Not front matter (content attributes)

Use these on image/video HTML or markdown attributes inside post content, not in front matter:

- `data-invert-dark="true"`
- `data-transparent-dark="true"`
- `data-no-invert`

## [Alerts](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#alerts)

Unified callout block. Usage: `{% include callout.html type="note" content="..." %}`
Types: info & note ("#0969da"), alert & warning ("#bf8700"), tip ("#1a7f37), important ("#8250df")

## Convert jupyter notebooks to posts

```
jupyter nbconvert --to markdown "<name.ipynb>" --output-dir _posts --output "<name>.md" --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True
```