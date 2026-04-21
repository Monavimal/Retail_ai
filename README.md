# Retail_ai
## Power BI Dashboard
![Retail Analytics Dashboard](dashboard.png)


Project Overview: A production-level Retail Analytics system.

Backend: Flask API deployed on Render.

AI Logic: Random Forest for Churn and TextBlob for Sentiment.

Visualization: Power BI Dashboard connected to AI-enhanced data.

Actionable Insight: Identified that customers with >90 days of inactivity have an 80% churn probability.


Final Recommendations & Insights:
Based on the machine learning model deployed at retail-ai-kf7h.onrender.com and the Power BI dashboard analysis, 
here are the key business recommendations:

1. Retention Strategy for High-Risk CustomersInsight: The model identified a high churn probability ($Churn = 1$) for customers who have not made a purchase in over 90 days.Action: Implement an automated Win-Back Email Campaign. Offer a personalized "We Miss You" discount code (e.g., 20% off) to any customer who hits the 85-day inactivity mark to prevent them from hitting the 90-day churn threshold.
   
2. Sentiment-Driven Product ImprovementsInsight: Sentiment analysis shows a correlation between neutral/negative reviews and lower total revenue per customer.Action: Task the product team to review items with a SentimentScore below 0.0. Focus on the top 5 most frequent complaints to improve product quality or description accuracy, as these are likely drivers of churn.

 3. Upselling Top-Performing ProductsInsight: Using the Recommendation System (Co-occurrence matrix), we identified specific products that are frequently bought together.Action: Update the e-commerce "Cart" page to suggest these correlated products in real-time. For example, if a customer adds Product A, immediately suggest Product B to increase the Average Order Value (AOV).

 4. Continuous Model MonitoringInsight: Retail trends change seasonally.Action: Schedule a monthly "Model Retraining" session using the latest data from retail_data.csv to ensure the Random Forest classifier remains accurate as customer behavior evolves.
