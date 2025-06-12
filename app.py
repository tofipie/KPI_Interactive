import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy import spatial

st.title("Interactive KPI and Text Similarity app 💬")

st.sidebar.title("App Description")

var_mapping = {'קבוצת מוצר':'MaterialGroup',
               'דמיון טקסטואלי':'Similarity_Category',
               'התאמה מדויקת - כמות פריטים':'Qty_Mismatch_Category',
               'התאמה מדויקת - מחיר':'Price_Mismatch_Category',
               'שם ספק':'VendorName'
    
}
    
 custom_names = list(var_mapping.keys())
 selected_custom_name = st.sidebar.selectbox('בחר מסמך', ['', *custom_names])
 selected_actual_name = var_mapping.get(selected_custom_name)

st.write('חישוב דמיון טקסטואלי')
# Load Transformers model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

user_input1 = st.text_input("תיאור פריט פנימי")
user_input2 = st.text_input("תיאור פריט ספק")
button = st.button("חשב דמיון טקסטואלי")

if user_input1 and user_input2 and button:
    emb1 = model.encode(user_input1)
    emb2 = model.encode(user_input2)
    similarity_score = 1 - spatial.distance.cosine(emb1, emb2)
    st.write(f"Similarity Score: similarity_score")


#variables = ['קבוצת מוצר','דמיון טקסטואלי','התאמה מדויקת בין כמות פריטים','התאמה מדויקת מחיר בחשבונית']
#Variables = st.selectbox("בחר פרמטר להצגת שיעורי KPI ", variables)

files = ['purchase_orders','goods_receipts','vendor_invoices',
'material_master','vendor_master','invoice_approvals']

with st.sidebar:
    st.write('שיעורי אישור חשבוניות לפי פרמטרים לבחירה')
    st.write("קבצים שנמצאים ב DB:")
    for file in files:
        st.markdown("- " + file)  
    st.write('Made by Noa Cohen')

report_df = pd.read_csv('report_df.csv')

titles_dict = {'MaterialGroup':'Approval Rate by MaterialGroup',
'Similarity_Category':'Approval Rate by Text Similarity Score',
'Qty_Mismatch_Category':'Approval Rate by Quantity',
'Price_Mismatch_Category':'Approval Rate by Price',
'VendorName':'Approval Rate by VendorName'}


title = titles_dict.get(selected_actual_name)

def generate_kpi_report(df, group_col, target_col='IsApproved', title=""):

    kpi_df = df.groupby(group_col)[target_col].agg(
        Total_Lines=('size'),
        Approved_Lines=('sum'),
        Rejection_Lines=(lambda x: (x == 0).sum())
    ).reset_index()
    kpi_df['Approval_Rate'] = (kpi_df['Approved_Lines'] / kpi_df['Total_Lines']) * 100
    kpi_df = kpi_df.sort_values(by='Approval_Rate', ascending=False)

    # Visualization
    fig = plt.figure(figsize=(3, 3))
    sns.barplot(x='Approval_Rate', y=group_col, data=kpi_df, palette='viridis')
    plt.title(title)
    plt.xlabel('Approval Rate')
    plt.ylabel(group_col)
    plt.show()
    st.pyplot(fig) # instead of plt.show()
    return kpi_df

kpi_result = generate_kpi_report(report_df,selected_actual_name,title = title)
st.write(kpi_result)


