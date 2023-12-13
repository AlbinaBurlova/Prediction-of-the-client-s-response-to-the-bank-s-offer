import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
import re

st.set_page_config(page_icon='🏦')

df = pd.read_csv("data.csv")
data = df.drop(columns=['ID_объекта'])

model = joblib.load('model.pickle')
scaler = joblib.load('scaler.pickle')
ordinal_encoder = joblib.load('encoder.pickle')

img = Image.open("bank_image.jpg")

st.title("Потенциальный отклик клиента на предложение банка")
st.image(img)
st.header("Разведочный анализ данных по откликам клиентов на предложения банка")
st.subheader("В этом приложении мы исследуем данные о клиентах банка, которые делали или не делали отклик на предложения банка. Данные содержат следующие признаки:")
st.write(df.columns.tolist())

st.write("Вот первые пять строк данных:")
st.write(df.head())
st.markdown("---")
st.write("Для начала посмотрим на распределение некоторых признаков.")

plt.figure(figsize=(10, 5))
palette = sns.color_palette("viridis", as_cmap=True)
sns.histplot(data=data, x='Возраст', bins=40, color=palette(0.6), kde=True)

mean_age = data['Возраст'].mean()
plt.axvline(mean_age, color='darkslateblue', linestyle='--')
plt.title(f'Распределение клиентов по возрасту')
st.pyplot(plt)
st.write("Средний возраст для клиента из выборки для анализа - 40 лет.")
st.markdown("---")
st.write("Отдельно посмотрим на распределение показателей целевой переменной.")

df['Целевая_переменная'] = df['Целевая_переменная'].replace({0: 'Не было отклика', 1: 'Был отклик'})
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Целевая_переменная', palette='viridis')
plt.title('Распределение целевой переменной')
st.pyplot(plt)
st.write("Стоит отметить значительный дисбаланс в распределении целевой переменной: из 15 тысяч объектов - 13 тысяч клиентов не откликнулись на предложение банка, и только чуть менее двух тысяч человек сделали отклик на услугу банка. Такое распределение в дальнейшем повлияет и на предсказательную модель.")
st.markdown("---")

df['Пол'] = data['Пол'].replace({0: 'Женщина', 1: 'Мужчина'})
df['Статус_работника'] = data['Статус_работника'].replace({0: 'Неработающий', 1: 'Работающий'})
df['Статус_пенсионера'] = data['Статус_пенсионера'].replace({0: 'Не является пенсионером', 1: 'Является пенсионером'})
df['Наличие_квартиры'] = data['Наличие_квартиры'].replace(
    {0: 'Нет квартиры', 1: 'Есть квартира'})

st.write("Бегло взглянем и на распределение других признаков:")
columns = ['Пол', 'Семейный_статус', 'Количество_детей', 'Количество_иждивенцев',
   'Статус_работника', 'Статус_пенсионера', 'Наличие_квартиры',
   'Собственный_автомобиль', 'Ссуды_клиента', 'Погашенные_ссуды',
   'Семейный_доход']

plt.figure(figsize=(10, len(columns) * 5))

for i, column in enumerate(columns):
    plt.subplot(len(columns), 1, i + 1)
    sns.countplot(data=df, x=column, palette='viridis')
    plt.title(f'{column}')

plt.tight_layout()
st.pyplot(plt)

st.markdown("---")
st.write("Ниже представлен корреляцинный график всех цифровых признаков.")

numeric_df = data.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица корреляций числовых признаков', fontsize=20, pad=20)
st.pyplot(plt.gcf())
st.write("Корреляционный график отражает одну важную особенность - среди числовых признаков и целевой переменной почти нет зависимостей. Наибольшая корреляция наблюдается с возрастом и доходом, но и она остается незначительной.")
st.write("Взглянем дополнительно на числовые характеристики распределения числовых столбцов:")

description = data.describe()
st.table(description.drop('count'))

st.write("Корреляционный анализ цифровых признаков показал, что на некоторые признаки и их корреляции с целевой переменной необходимо рассмотреть детальнее.")

plt.figure(figsize=(10, 5))
sns.boxplot(data=data, x='Целевая_переменная', y="Возраст", palette='viridis')
plt.title(f'Распределение признака "Возраст" для каждого класса целевой переменной')
st.pyplot(plt)
st.write("Можно заметить, что возраст клиента, который откликается на предложение банка в среднем немного ниже, чем у человека, который такое предложение игнорирует.")

contingency_table = pd.crosstab(data['Погашенные_ссуды'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 4))
sns.heatmap(percentage_distribution, annot=True, cmap='BuGn', fmt=".2f")
plt.title('Корреляции между погашенными ссудами и откликом (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())

st.write("Стоит игнорировать небольшой выброс на значении 8, пристальнее взглянув на людей без погашенных ссуд, они откликаются на предложения банка немного чаще других.")

bins = [0, 10000, 30000, 50000, 100000, 250000]
labels = ["до 10000 руб.", "от 10000 до 30000 руб.", "от 30000 до 50000 руб.", "от 50000 до 100000 руб.",
          "от 100000 до 250000 руб."]

data['Доход'] = pd.cut(data['Личный_доход'], bins=bins, labels=labels, include_lowest=True)

contingency_table = pd.crosstab(data['Доход'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 4))
sns.heatmap(percentage_distribution, annot=True, cmap='BuGn', fmt=".2f")
plt.title('Корреляции между доходом и откликом (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())

plt.figure(figsize=(10, 4))
sns.countplot(data=data, x="Доход", palette='viridis')
plt.title(f'Распределение личного дохода клиента')
plt.tight_layout()
st.pyplot(plt)

st.write("Распределение личного дохода и его корреляция с целевой переменной показывают, что с возрастанием дохода увеличивается и отклик клиентов на предложение банка.")

st.markdown("---")
st.write("Оценим корреляции между откликами клиентов на банковские услуги и отдельными категориальными признаками, такими как семейный доход, уровень образования, должность и семейный статус.")
st.write("Распределение представлено в процентах.")

# порядок значений в "Семейном доходе" почему-то перемешан, надо это исправить
income_order = ['свыше 50000 руб.', 'от 20000 до 50000 руб.', 'от 10000 до 20000 руб.', 'от 5000 до 10000 руб.', 'до 5000 руб.']
data['Семейный_доход'] = pd.Categorical(data['Семейный_доход'], categories=income_order, ordered=True)

contingency_table = pd.crosstab(data['Семейный_доход'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 4))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для семейного дохода (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())

st.write("Семейный доход отражает занимательную тенденцию - отклик значительно увеличен среди людей с семейным доход свыше 50000 рублей.")

income_order = ['Ученая степень', 'Два и более высших образования', 'Высшее', 'Неоконченное высшее', 'Среднее специальное', 'Среднее', 'Неполное среднее']
data['Образование'] = pd.Categorical(data['Образование'], categories=income_order, ordered=True)
contingency_table = pd.crosstab(data['Образование'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 5))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для уровня образования (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())

st.write("При оценке зависимости между уровням образования и целевой переменной, можно заметить, что около 20% из категории людей с двумя и более высшими образованиями и с неоконченным высшим откликаются на предложение банка.")

contingency_table = pd.crosstab(data['Должность'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 6))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для должности (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())
st.write("Анализируя распределение по должностям, можно заметить, что наибольший отклик на товар банка происходит от партнера (подразумевается, партнера в фирме), военнослужащего, индивидуального предпринимателя, а также руководителей высшего и низшего звена.")

st.write("Интересно отметить, что по сравнению с работающими гражданами пенсионеры почти не откликаются на предложения банка.")

contingency_table = pd.crosstab(data['Семейный_статус'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 4))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для семейного статуса (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())

st.write("Наибольший отклик наблюдается среди людей, состоящих в гражданском браке.")

st.markdown("---")

st.write("Поскольку тепловая корреляция должности и целевой переменной оказалась информативной, стоит дополнительно взглянуть на данные по отраслям работы клиента, а также по направлению деятельности клиента.")
contingency_table = pd.crosstab(data['Отрасль_работы'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 20))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для отрасли работы клиента (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())
st.write("При анализе корреляций между отраслями деятельности клиентов и откликом на услуги банка, можно выделить сферу недвижемости - 35% людей, занятых в сфере недвижемости, делают отклики на услуги банка. Следующий сектор - это сфера общественного питания и ресторанный бизнес.")

contingency_table = pd.crosstab(data['Направление_деятельности'], data['Целевая_переменная'])
percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
plt.figure(figsize=(10, 10))
sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта для направления деятельности (в процентах)', fontsize=15, pad=20)
st.pyplot(plt.gcf())
st.write("Но если мы смотрим на деятельность самых клиентов, а не на сферу занятости, то лидирующую позицию по откликам занимают люди, занимающиеся рекламой и маркетингом. Здесь стоит отметить, что на прошлом графике маркетинг был с нулевыми показателями - здесь важно отличать отрасли от самого направления деятельности клиента. Подразумается, что клиент может работать маркетологом, но в сфере недвижимости или общественного питания, а не в рекламной фирме.")
st.markdown("---")
st.subheader("Вывод")
st.write("Несмотря на большое количество объектов в анализируемых данных, несбалансированность по количеству значений целевой переменной ведет к снижению информативности этих данных. При этом незначительная корреляция между определенными количественными и качественными переменными может быть основой для вполне эффективной предсказательной модели.")

# Боковая панель
st.sidebar.title("Предсказать отклик клиента")

gender = st.sidebar.selectbox("Пол", ['Мужчина', 'Женщина'])
age = st.sidebar.slider("Возраст", 0, 100, 30)
education = st.sidebar.selectbox("Образование", ['Неполное среднее', 'Среднее', 'Среднее специальное', 'Неоконченное высшее', 'Высшее', 'Два и более высших образования', 'Ученая степень'])
marital_status = st.sidebar.selectbox("Семейный статус", ['Состою в браке', 'Гражданский брак', 'Разведен(а)', 'Не состоял в браке', 'Вдовец/Вдова'])
child_total = st.sidebar.slider("Количество детей", 0, 10)
dependants = st.sidebar.slider("Количество иждивенцев", 0, 10)
socstatus_work_fl = st.sidebar.selectbox("Статус работника", ['Работает', 'Не работает'])
socstatus_pens_fl = st.sidebar.selectbox("Статус пенсионера", ['Пенсионер', 'Не пенсионер'])
fl_presence_fl = st.sidebar.selectbox("Наличие квартиры", ['Есть', 'Нет'])
own_auto = st.sidebar.slider("Собственный автомобиль", 0, 2)
loan_num_total = st.sidebar.slider("Ссуды клиента", 0, 15)
loan_num_closed = st.sidebar.slider("Погашенные ссуды", 0, 15)
family_income = st.sidebar.selectbox("Семейный доход", ['до 5000 руб.', 'от 5000 до 10000 руб.', 'от 10000 до 20000 руб.', 'от 20000 до 50000 руб.', 'свыше 50000 руб.'])
personal_income = st.sidebar.number_input("Личный доход", min_value=0)
gen_industry = st.sidebar.selectbox("Отрасль работы", ['Торговля', 'Информационные технологии', 'Образование', 'Государственная служба', 'Другие сферы', 'Сельское хозяйство', 'Здравоохранение', 'Металлургия/Промышленность/Машиностроение', 'Коммунальное хоз-во/Дорожные службы', 'Строительство',
       'Транспорт', 'Банк/Финансы', 'Ресторанный бизнес/Общественное питание', 'Страхование', 'Нефтегазовая промышленность', 'СМИ/Реклама/PR-агенства',
       'Энергетика', 'Салоны красоты и здоровья', 'ЧОП/Детективная д-ть','Развлечения/Искусство', 'Наука', 'Химия/Парфюмерия/Фармацевтика',
       'Сборочные производства', 'Туризм', 'Юридические услуги/нотариальные услуги', 'Маркетинг', 'Подбор персонала', 'Информационные услуги', 'Недвижимость',  'Управляющая компания', 'Логистика', 'На пенсии', 'Другие сферы'])
gen_title = st.sidebar.selectbox("Должность", ['Рабочий', 'Специалист', 'Руководитель среднего звена',  'Руководитель высшего звена', 'Служащий', 'Работник сферы услуг', 'Высококвалифиц. специалист', 'Индивидуальный предприниматель', 'Военнослужащий по контракту', 'Руководитель низшего звена',
       'Другое', 'Партнер', 'На пенсии', 'Другое'])
job_dir = st.sidebar.selectbox("Направление деятельности", ['Вспомогательный техперсонал', 'Участие в основ. деятельности', 'Адм-хоз. и трансп. службы', 'Пр-техн. обесп. и телеком.',
       'Служба безопасности', 'На пенсии', 'Бухгалтерия, финансы, планир.', 'Снабжение и сбыт', 'Кадровая служба и секретариат', 'Юридическая служба',
       'Реклама и маркетинг'])
work_time = st.sidebar.number_input("Время работы на последнем рабочем месте (в месяцах)", min_value=0)

button = st.sidebar.button('Получить предсказание!')

if button:

    input_values = [age, gender, education, marital_status, child_total, dependants, socstatus_work_fl, socstatus_pens_fl, fl_presence_fl, own_auto, loan_num_total, loan_num_closed, family_income, personal_income, gen_industry, gen_title, job_dir, work_time]
    columns = ['AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'OWN_AUTO', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'FAMILY_INCOME', 'PERSONAL_INCOME', 'GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR', 'WORK_TIME']
    input_df = pd.DataFrame([input_values], columns=columns)

    input_df['GENDER'] = input_df['GENDER'].replace({'Женщина': 0, 'Мужчина': 1})
    input_df['SOCSTATUS_WORK_FL'] = input_df['SOCSTATUS_WORK_FL'].replace({'Работает': 1, 'Не работает': 0})
    input_df['SOCSTATUS_PENS_FL'] = input_df['SOCSTATUS_PENS_FL'].replace({'Пенсионер': 1, 'Не пенсионер': 0})
    input_df['FL_PRESENCE_FL'] = input_df['FL_PRESENCE_FL'].replace({'Есть': 1, 'Нет': 0})

    def replace_values(x):
        if "от" in x and "до" in x:
            numbers = [int(n) for n in re.findall(r"\d+", x)]
            return sum(numbers) / len(numbers)
        else:
            number = re.search(r"\d+", x).group()
            return int(number)


    input_df['FAMILY_INCOME'] = input_df['FAMILY_INCOME'].apply(replace_values)

    bins = [0, 10000, 30000, 50000, 100000, 250000]
    labels = ["до 10000 руб.", "от 10000 до 30000 руб.", "от 30000 до 50000 руб.", "от 50000 до 100000 руб.",
              "от 100000 до 250000 руб."]

    input_df['INCOME'] = pd.cut(input_df['PERSONAL_INCOME'], bins=bins, labels=labels, include_lowest=True)
    input_df = ordinal_encoder.transform(input_df)

    input_df = scaler.transform(input_df)

    probs_test = model.predict_proba(input_df)
    prediction = probs_test[:, 1] >= 0.5

    response = "Отклик!" if prediction else "Отклика нет."

    st.sidebar.write(f'Предсказание модели: {response}')

    response = "Отклик!" if prediction else "Отклика нет."

    st.sidebar.write(f'Предсказание модели: {response}')
