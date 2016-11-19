# Machine Learning Engineer Nanodegree
## Capstone Proposal 
Gil Akos 
November 4th, 2016

## Proposal // Financial Transaction Predictor

### Domain Background
Managing your personal finances is a time-consuming and stress-inducing activity. With the rise of services and apps that mark the trend towards the unbundling of banks [^first], the "complicated-ness" of personal finance has only increased and our ability as individuals to optimally manage our expanding number of accounts has only become more challenging. From a systemic viewpoint the complexity of banking is also significant - more than $14 trillion moves flows through the banking system in the United States every day [^second]. Furthermore, the transactions that make up these flows are often inconsistently formatted or minimally labeled. Even large fintech aggregator apps do a poor job of munging, cleaning, and categorizing transaction data. Here are a few hilarious examples from my own Mint account (which I do manage with some regularity):

`$9.99 PMG CLE AIRPRT 10/25 #000322122 PURCHASE 18930 BROOKPARK R CLEVELAND OH` becomes **PMG CLE AIRPRT** categorized as **Doctor**
*I hope I didn't go to a discount Doctor at the Cleveland Airprt (mis-spelling intentional).* 

`$29.05 BOGARTS SMOKEHOUSE 04/10` becomes **BOGARTS** categorized as **Personal Care**
*I tried hard, but am having a difficult time rationalizing how delicious St. Louis Style Bar-B-Que helps me with my personal care.*

As we can see in the two examples above, we would receive at a minimum three data points per transaction - Amount, Date, and a Textual Description. Implictly we also have which account the transaction processes through, as well as day of the week, day of the month, and day of the year. Building a model using deep learning with the capability of predicting categorization and future cash flow would have value in that it would help us optimize the work our money does for us while reducing the stress of having to do so directly and explicitly[^third]. This transaction data is key - but how many dimensions do we need, and can we infer labels i.e. categories when the data is lacking detail (as we can directly see in the second example above)? 

Across the fintech industry, there are some relatively popular players in the aggregation, budgeting, and forecasting game. Mint[^fourth] pulls data from as many different accounts that you connect and tries to clean and categorize the transactions (with relative success - see above), Level Money[^fifth] offers up projections of your cash flow and spending plotted across the month (overly coarse in detail in my experience), and Plaid[^sixth] offers an API that formats transaction data into digestible formats, even predicting some labels with a paired confidence value to indicate reliability (label filling only works for some transaction types). Outside of the industry activity around this problem, there are also plenty of articles on using Deep Learning using Long Short-Term Memory models for creating predictions with financial time series data (list below).

I am particularly passionate about this problem becuase not only am I frustrated by the lack of smarts in the tools I use to manage my finances, but also this is a key ingredient to the next generation of technology we are building for my startup, Astra. Creating models that can accurately predict one of our key labels given the other two would beneficial to our progress as well as the general studies that seek to use deep learning for financial predictions.

- [Deep Learning for Time Series Modeling](http://cs229.stanford.edu/proj2012/BussetiOsbandWong-DeepLearningForTimeSeriesModeling.pdf)
- [Deep Learning for Multivariate Financial Time Series](http://www.math.kth.se/matstat/seminarier/reports/M-exjobb15/150612a.pdf)
- [Machine Learning with Financial Time Series Data](https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data)
- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [TensorFlow Tutorial for Time Series Prediction](https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series)


[^first] [TechCrumnch // What’s next for personal financial services?](https://techcrunch.com/2016/04/23/whats-next-for-personal-financial-services/)
> If 2015 was the year of the great “Bank Unbundling,” with new companies dissecting the consumer banking experience to offer specialized services, it was also a year that saw the emergence of a new landscape of financial influencers taking a seat at the table.

[^second] [Federal Reserve Bank of New York // Intraday Liquidity Flows](http://libertystreeteconomics.newyorkfed.org/2012/08/intraday-liquidity-flows.html)
> On a typical day, more than $14 trillion of dollar-denominated payments is routed through the banking system.

[^third] [TechCrunch // AI can make your money work for you] (https://techcrunch.com/2016/09/08/ai-can-make-your-money-work-for-you/)
>  Did you know that extra cash in your checking account is a missed opportunity? Every day, it loses value to inflation. To generate better returns, you could keep the bare minimum in your checking account and invest the rest. However, unexpected expenses can drain your account suddenly. Without extra cushioning in your checking account, you risk getting slapped with bank fees or credit card debt that quickly cancel out any gains from your investments. It feels like you can’t win.  Either you’re missing out on capital gains, or you’re playing limbo with your account balance. AI will make this struggle a thing of the past. Advances in AI will create a robo-accountant that knows your spending better than you do. By analyzing your purchase history, it will constantly move money between your checking, savings, investments and credit cards. This way, your checking account’s balance is always in the narrow “sweet spot:” high enough to avoid fees, but not so high that you miss out on investment yield.

[^fourth] [Mint](https://www.mint.com/)

[^fifth] [Level Money](https://www.levelmoney.com/)

[^sixth] [Plaid](https://plaid.com/)

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

Personal Fin Data
App to allow individuals to connect to mint

Parameters of data
- Date
- Amount
- Description
- Implied
-- Account
-- Day of Week
-- Day of Month
-- Day of Year

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

Tensorflow
- Long Term Short Memory
- Support Vector Machines

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

### [Footnotes]


-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?