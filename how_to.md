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

  In markdown, add `{: data-invert-dark="true"}` after the image

- Prevent auto inversion (automatically checks if >70% image is white or transparent)

  `<img src="..." data-no-invert>`

- Light card background in dark mode (for diagrams with text on white bg, GIFs, videos): `{: data-transparent-dark="true"}`

- To manually remove white background, run `python _scripts/white2transparent.py "images\multi-task-methods.jpeg" --webp --max-width 1000 --feather 55 --threshold 255`. The same script will also convert to webp format which has much lower image sizes.

## [Alerts](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#alerts)

Unified callout block. Usage: `{% include callout.html type="note" content="..." %}`
Types: info & note ("#0969da"), alert & warning ("#bf8700"), tip ("#1a7f37), important ("#8250df")

## Convert jupyter notebooks to posts

```
jupyter nbconvert --to markdown "<name.ipynb>" --output-dir _posts --output "<name>.md" --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True
```