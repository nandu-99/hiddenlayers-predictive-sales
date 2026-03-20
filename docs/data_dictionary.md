# Data Dictionary

This document details the schema of the datasets used in the predictive sales project. The primary datasets are stored externally and should be downloaded into the `data/raw/` directory.

## 1. SaaS Sales Conversations

**Source:** [DeepMostInnovations/saas-sales-conversations](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations)

A synthetic dataset designed for training sales conversion prediction models, containing realistic dialogues between sales representatives and potential customers.

| Column Name | Data Type | Description | Sample/Format |
| :--- | :--- | :--- | :--- |
| `company_id` | String/Integer | A unique identifier for the SaaS company. | "C-12345" |
| `company_name` | String | The name of the SaaS company. | "TechFlow Solutions" |
| `product_name` | String | The name of the SaaS product. | "FlowCRM Core" |
| `product_type` | String | The industry or category of the product. | "CRM Software" |
| `conversation_id` | String | A unique identifier for each conversation. | "CONV-98765" |
| `scenario` | JSON | A JSON object containing details about the conversation scenario. | `{"goal": "demo", "role": "buyer"}` |
| `conversation` | JSON Array | A JSON array of conversation messages. | `[{"speaker": "sales", "text": "Hi"}]` |
| `full_text` | String | The complete text of the conversation. | "Sales: Hi there... Customer: Hello." |
| `outcome` | Integer | A binary value indicating the conversion outcome (0 for no conversion, 1 for conversion). | `1` |
| `conversation_length` | Integer | The number of messages in the conversation. | `15` |
| `customer_engagement` | Float | A score (0-1) representing customer engagement. | `0.85` |
| `sales_effectiveness` | Float | A score (0-1) representing sales representative effectiveness. | `0.92` |
| `probability_trajectory` | JSON | A JSON object showing the conversion probability at each turn of the conversation. | `{"turn_1": 0.1, "turn_5": 0.4}` |
| `conversation_style` | String | The style of the conversation. | "casual_friendly", "direct_professional"|
| `conversation_flow` | String | The flow pattern of the conversation. | "standard_discovery" |
| `communication_channel`| String | The channel used for communication. | "email", "phone", "chat" |
| `embedding_*` | Float | 3072-dimensional embedding vector (columns `embedding_0` to `embedding_3071`). | `[0.012, -0.045, ...]` |

---

## 2. CRM Sales Opportunities

**Source:** [Maven Analytics CRM Sales Opportunities](https://mavenanalytics.io/data-playground/crm-sales-opportunities)

A B2B sales pipeline dataset originating from a fictitious computer hardware company. 

| Column Name | Data Type | Description | Sample/Format |
| :--- | :--- | :--- | :--- |
| `sales_agent` | String | The sales representative managing the opportunity. | "Anna Vallejo" |
| `product` | String | The specific hardware product involved in the sale. | "GTX Basic" |
| `account` | String | The company or client targeted for the sale. | "Kroger" |
| `deal_stage` | String | The current phase of the sales process. | "Prospecting", "Engaging", "Won", "Lost"|
| `engage_date` | Date | The date when the engagement stage began. | `2017-10-20` |
| `close_date` | Date | The date when the deal was finalized (won or lost). | `2017-11-05` |
| `close_value` | Float | The revenue generated from the deal. | `1250.00` |
