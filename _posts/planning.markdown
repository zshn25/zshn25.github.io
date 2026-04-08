

You sit in your car, open Navigator and type your destination. The map plans a route within seconds. For autonomous driving though, this is not enough. Navigator route is coarse and does not include a lot of information such as lanes, other road-users (vehicles, pedestrians, bicycles, etc.), traffic-lights and so on. Sure, some advanced navigators also provide HD-maps: fine-grained precise map information including road-markings, traffic signs, etc. As you may imagine, maintaining these high accuracy maps is a big challange: the lanes, speed-limits on multiple roads change everyday due to maintanance, repair or accident. Google dencentralized this and makes use of it's Map netowrk by asking users about roads. They are able to to do this since Google MAps is already the most widely used navigator. They also started offering HD maps to auto manufacturers [^1,2]. Other HD-Map service (MaaS: Map as a Service) providers include https://www.here.com/platform/adas-had, 

(Maybe make a post on maps). Check https://www.thinkautonomous.ai/blog/robot-mapping/

Navigator info comes from map services which build a static map of our world but driving in the real-world requires also adressing the dynamic objects. This is where 

After working on perception all my life, for the past 2 years, I got the oppertunity to work on the next part of Autonomous Driving, namely prediction and planning

World -> Sensor -> Perception -> Useful intermediates -> Prediction & Planning -> Control 

Planning relies on perception to produce rich representations of the world. It needs (HD-maps with traffic-lights, lanes, crosswalks, etc.; all vehicles, pedestrians tracked for a long time in many scenarios). Most of these come from perception outputs (detecting other vehicles, pedestrians and tracking them over time, traffic light detection, traffic sign classification, etc.). Some my either come from perception or map providers (lane boundaries, crosswalks, stop-lines, speed limit)

Planner papers:

- EPSILON: An Efficient Planning System for Automated Vehicles in Highly Interactive Environments [Paper](https://arxiv.org/abs/2108.07993), [Code](https://github.com/HKUST-Aerial-Robotics/EPSILON)
- PDM 
 - When PDM's internal predictor (constant velocity) is changed to GT, it does not lead to higher performance, indicating that there is no scope for improvement in this direction. Perfect Prediction or Plenty of Proposals? What Matters Most in Planning for Autonomous Driving. [Paper](https://arxiv.org/abs/2510.15505)

World Models (Generative Video Action models for End-to-End)
- Generateive Transformers
    - ![drivinggpt](https://rogerchern.github.io/DrivingGPT/static/images/drivinggpt_teaser.png) [DrivingGPT](https://rogerchern.github.io/DrivingGPT/), [Paper](https://rogerchern.github.io/DrivingGPT/static/pdfs/drivinggpt.pdf), [Code](https://github.com/RogerChern/DrivingGPT) generates future videos and planned actions

Evaluation:

It is difficult to evaluate planners due to their extensive data requirements and thier inherent behaviour, changing the environment itself. 


Acquiring such data is very difficult and even more difficult to scale (As mentioned in my talk, this is the disadvantage of modular approach. End-to-end methods are relatively easy to collect data). Companies like Waymo's most value lies in collecting these. Till recently such data was closed and only openly available options were Argoverse (2019: 570 hours), Lyft Level 5 data (2019: 1000+ hours), Waymo Open Dataset (2021: 570 hours). In all these benchmarks, it was only possible to do Open Loop (planner doesn't drive the vehicle, it's output is only compared to human driver's GT). NuPlan was the first benchmark to introduce closed-loop real world benchmarking with 1200 hours (Check nuPlan for more info). Motional had parts of this dataset released before (nuScenes, nuImages) but the nuPlan combines them all for what is required to predict and plan.


- [![pred2plan](https://github.com/aumovio/pred2plan/raw/main/assets/Pred2Plan.png)](https://arxiv.org/pdf/2505.05638) Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks — [Paper](https://arxiv.org/pdf/2505.05638), [Code](https://github.com/aumovio/pred2plan)
    - incorporates many prediction and planning algorithms in nuPlan benchmark
- 








[^1]: https://www.volvocarstoronto.com/en/news/view/exploring-google-hd-maps-in-the-brand-new-volvo-ex90/124798#:~:text=This%20is%20where%20Google%20HD,for%20advanced%20driving%20assistance%20systems.&text=Developed%20specifically%20for%20car%20manufacturers,constantly%20updated%20representation%20of%20roadways.

[^2]: https://techcrunch.com/2023/01/05/google-hd-maps-volvo-polestar/

https://www.mathworks.com/help/driving/ug/use-here-hd-live-map-data-to-verify-lane-configurations.html