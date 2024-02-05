## MCM2024-C Code

**Team 2422433**

In the 2023 Wimbledon Men’s Final, Djokovic’s early lead was overturned by Alcaraz, who clinched victory in a dynamic final set, revealing the match’s momentum and swings. This paper introduces three cascading models to analyze and forecast such changes.

Firstly, we developed a Nonlinear Weighted AHP Model to assess players’ performance, structured into three levels - Objective, Criterion, and Plan, focusing on four primary indicators - Point Victory, Tension, Technique, and Physical Exertion, plus secondary indicators. Applied to the 2023 Wimbledon Men’s Final, its effectiveness was confirmed by comparing model outcomes with actual match conditions.

Secondly, in our study on momentum in tennis, we assessed its presence by comparing it to random occurrences, focusing on two key indicators: Confidence and Performance. Confidence was calculated using weighted technic and score metrics, and the influence of momentum was modeled with exponential decay. Analyzing the 2023 Wimbledon Gentlemen’s Final, we found significant Pearson correlation coefficients (0.223 and 0.227) for both players, contrasting with a random scenario (coefficient of 0). This suggests that momentum is a factor in tennis matches.

Thirdly, to predict momentum flow and swing occurrence, we developed an improved LSTM Model. To define a swing, we used fluctuation in the difference between two players’ momentum as a criterion, based on mainstream media and sports commentary. Subsequently, we achieved prediction of swings through the model. We enhanced the traditional LSTM model by incorporating an output block for predictions, which has a dropout layer to prevent over-fitting. Using data for pre-training, we created a Pre-trained Model. This model was then fine-tuned with the preceding data of the test match, effectively addressing the issue of limited data and accurately forecasting match swings.

Finally, we tested the improved LSTM model’s universality and robustness on four given matches and various other tennis tournaments. It showed good predictive performance for Wimbledon Men’s matches and moderate predictive ability elsewhere. However, performance varied due to differences in match formats, players’ adaptability to court surfaces, and different sports’ properties. We recommend enhancing the model by adjusting the factors and boosting its learning efficiency for better predictions.

Keywords: tennis, AHP, LSTM, momentum, swing.