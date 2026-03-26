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

- Transparent background (white → transparent): `{: data-transparent-dark="true"}`

## [Alerts](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#alerts)

Unified callout block. Usage: `{% include callout.html type="note" content="..." %}`
Types: info & note ("#0969da"), alert & warning ("#bf8700"), tip ("#1a7f37), important ("#8250df")
