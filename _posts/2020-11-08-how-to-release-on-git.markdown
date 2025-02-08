---
layout: post
title:  "How to Release on Github"
description: "Create releases of your repository and deliver versions of code deployments on GitHub"
image: https://octodex.github.com/images/octonaut.jpg
date:   2020-11-08 18:44:35 -0500
categories: github tags releases
author: Zeeshan Khan Suri
published: true
comments: true
---



To release on Git, you need to tag a commit first.

## Tagging a commit

A tag is a label attached to a specific commit. There are two types of tags

- Lightweight
- Annotated

`git tag` command is used to view all tags in the repo.

### Lightweight tag

The lightweight tag is a simple reference to a commit.

`git tag <tagname> <commit>` creates a lightweight tag. If `<commit>` is optional and defaults to `HEAD`.

#### Example

```bash
$ git tag v0.1 HEAD^ # tag the previous commit
$ git tag # view all tags
v0.1
$ git tag v1.0 # tag the current commit
$ git tag # view all tags
v0.1
v1.0
```

### Annotated tag

The annotated tag is a full git object which includes additional (annotated) information such as tag author info, tag date, tag message and tag commit ID. In general, annotated tags are recommended over lightweight

To tag a commit with an annotated tag, use the `git tag` command with `-a` option. You must also specify a message using `-m` option

#### Example

```bash
$ git tag -a -m "feature release 1.0" v1.0
```

### Pushing tags

`git push` does not automatically push the tags to remote repository. 

- To transfer a single tag, use 
  - `git push <remote> <tagname>`
- To transfer all tags, use
  - `git push <remote> --tags`

## Releases

Once you tag a commit, you can use this to create a release. On [GitHub](https://github.com), you can create a release by following [this tutorial](https://docs.github.com/en/free-pro-team@latest/github/administering-a-repository/managing-releases-in-a-repository "Creating a release") 