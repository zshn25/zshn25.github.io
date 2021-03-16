---
layout:     post
title:      "Guide to Ethereum Mining"
description: "Without paying any fees"
comments: true
date:       2021-01-12 15:21:23 -0700
categories: cryptocurrency ethereum 
author:     Zeeshan Khan Suri
hidden:    true
hide: true
search_exclude: true
---

{::options parse_block_html="true" /}

___

A very simple and straightforward way to earn money while lending your computing power is to use [![NiceHash](https://www.nicehash.com/static/logos/logo_big_light.png){:height="36px"}](https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4). If you are just starting to mine and looking for an easy way in, then this is it.  This does not primarily mine cryptocurrency for you but this will lend your computer for others to mine cryptocurrencies. You will be paid in Bitcoins for lending your computer. [NiceHash](https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4) allows you to lend your computing hash power to other users. Other users pay you for your hash power but since [NiceHash]((https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4)) is the intermediatory, it charges fees for it's service, which is mentioned [here](https://www.nicehash.com/support/general-help/nicehash-service/fees). [[*<sub><sup>Disclaimer</sup></sub>*]](#disclaimer-this-is-a-referral-link-where-i-benefit-if-you-spend-money-on-their-website)

[![](https://www.nicehash.com/img/mining_overview.db925858.gif)](https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4)
{:refdef: style="text-align: center;"}
*Fig.1: How [NiceHash](https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4) works*
{: refdef}

___

## Mine yourself

For advanced users who wish to mine themselves, read further. In this guide, I mention how to mine [Ether](https://ethereum.org/en/eth/) (ETH). If you want to learn more about what Ethereum mining is, I would like to refer you to [this simple but exhaustive guide](https://eth.wiki/en/fundamentals/mining). Ether is the currency of [Ethereum](https://ethereum.org/en/what-is-ethereum/). I chose Ether because right now it is the most profitable cryptocurrency to mine according to [What to mine](https://whattomine.com/coins). 

## Requirements

- Computer with an NVIDIA/AMD GPU with at least 4Gb of [video RAM memory](https://en.wikipedia.org/wiki/Video_RAM_(dual-ported_DRAM)). If you don't have it, then I am sorry, this is not for you. You can still use [![NiceHash](https://www.nicehash.com/static/logos/logo_big_light.png){:height="36px"}](https://www.nicehash.com/?refby=88408201-970f-4256-9837-796807b14ba4) though.

- Ethereum wallet (To store your hard earned ETH). [Find one yourself](https://ethereum.org/en/wallets/find-wallet/) if your don't already have one.

- Mining software.

- Mining pool. (Because it is almost impossible to mine alone now-a-days.)

### Mining software

The top mining softwares are 

- [Ethminer](https://github.com/ethereum-mining/ethminer)
- [T-Rex](https://github.com/trexminer/T-Rex)
- [Claymore](https://claymoredualminer.com/)
- [Pheonix](https://phoenixminer.org/)

Except Ethminer, all other softwares charge a 1% dev fee. That means, 1% of all our mined ETH goes to the developer. The comparison of these softwares is out of scope of this article but feel free to choose your own. The following procedure would still remain similar

### Mining pools

Mining ETH alone is almost impossible at this moment. That it why, it is advised to mine with other miners in a pool. The profit at the end will be shared among the participants of the pool. For a list of currently active pools, visit [Poolwatch](https://www.poolwatch.io/coin/ethereum). In general choose the pool with the highest hash rate but also keep in mind [the payout method](https://en.wikipedia.org/wiki/Mining_pool#Mining_pool_methods), and minimum payouts and pool fees. For example, [Ethermine](https://www.ethermine.org/) has a min payout of 0.05 ETH while others have 0.1 ETH. It could take you twice the amount of mining time to get your hard earned ETH onto your wallet if you chose the latter.

![poolwatch.io](/assets/miningpools.PNG)
{:refdef: style="text-align: center;"}
*Fig.2: Metric in paranthesis () is the dev fee. [Source: Poolwatch](https://www.poolwatch.io/coin/ethereum)*
{: refdef}

## Example

I chose Ethminer because of no dev fee and it being relatively simple. The instructions for Ethminer are mentioned in the [Ethminer's guide](https://github.com/ethereum-mining/ethminer/blob/master/docs/POOL_EXAMPLES_ETH.md) but I will simplify them as follows.

- Download Ethminer and unzip in <Ethminer directory>

- Open Terminal/Command Line in <Ethminer directory> and do

<!-- ``` -->
{% highlight bash linenos %}
setx GPU_FORCE_64BIT_PTR 0
setx GPU_MAX_HEAP_SIZE 100
setx GPU_USE_SYNC_OBJECTS 1
setx GPU_MAX_ALLOC_PERCENT 100
setx GPU_SINGLE_ALLOC_PERCENT 100
ethminer.exe --farm-recheck 200 <-U if using NVIDIA or -G if using AMD GPU> -P stratum://<enter-your-wallet-address-here>.<give-worker-a-name>@<pool-address>:<port-number>
{% endhighlight %}
<!-- ``` -->



Head to your pool to get the pool address and the port number. For example, for [Ethermine pool](https://www.ethermine.org/start), and my wallet address, this becomes

```
ethminer.exe --farm-recheck 200 -U -P stratums://0x5E16Af66709efE1903065C5d4f9617ffD8C28714.zee@eu1.ethermine.org:5555 -P stratum://0x5E16Af66709efE1903065C5d4f9617ffD8C28714.zee@eu1.ethermine.org:4444
```

Notice that Ethermine pool charges 1% as pool fee. You could also choose [Hiveon pool](https://hiveon.net/) which doesn't charge any fee but as seen in Fig 2., since it's hashrate is too low. I chose Ethermine pool not only because of good hashrate but also of the payout policy. The min payout is just 0.05 ETH (in practise, it's even less), so you will get your hard mined ETH early.

Tip: After starting mining, head to your pool website and enter your Wallet address. It will show you all your workers' details and lets you track your payouts. You can also set your minimum payout there.


### Happy mining!




###### *Disclaimer: This is a referral link where I benefit if you spend money on their website*


:::info
This is a alert area.
:::