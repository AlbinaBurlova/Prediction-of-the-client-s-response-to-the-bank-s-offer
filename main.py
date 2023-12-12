# Импортируем библиотеки
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

df = pd.read_csv("data.csv")

img = Image.open("bank_image.jpg")

pages = ["Разведочный анализ данных", "Предсказать отклик по данным"]

page = st.sidebar.radio("Выберите страницу", pages)

if page == "Разведочный анализ данных":
    st.title("Потенциальный отклик клиента на предложение банка")
    st.image(img)
    st.header("Разведочный анализ данных по откликам клиентов на предложения банка")
    st.subheader("В этом приложении мы исследуем данные о клиентах банка, которые делали или не делали отклик на предложения банка. Данные содержат следующие признаки:")
    st.write(df.columns.tolist())

    st.write("Вот первые пять строк данных:")
    st.write(df.head())
    st.markdown("---")
    st.write("Ниже представлен корреляцинный график всех цифровых признаков.")

    data = df.drop(columns=['ID_объекта'])
    numeric_df = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Матрица корреляций числовых признаков', fontsize=20, pad=20)
    st.pyplot(plt.gcf())
    st.write("Корреляционный график отражает одну важную особенность - среди числовых признаков и целевой переменной почти нет зависимостей. Наибольшая корреляция наблюдается с возрастом и доходом, но и она остается незначительной.")
    st.markdown("---")
    st.write("Оценим корреляции между откликами клиентов на банковские услуги и отдельными категориальными признаками, такими как семейный доход, уровень образования, должность и семейный статус.")
    st.write("Распределение представлено в процентах.")

    # порядок значений в "Семейном доходе" почему-то перемешан, надо это исправить
    income_order = ['свыше 50000 руб.', 'от 20000 до 50000 руб.', 'от 10000 до 20000 руб.', 'от 5000 до 10000 руб.', 'до 5000 руб.']
    data['Семейный_доход'] = pd.Categorical(data['Семейный_доход'], categories=income_order, ordered=True)

    contingency_table = pd.crosstab(data['Семейный_доход'], data['Целевая_переменная'])
    percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Тепловая карта для семейного дохода (в процентах)', fontsize=15, pad=20)
    st.pyplot(plt.gcf())

    st.write("Семейный доход отражает занимательную тенденцию - отклик значительно увеличен среди людей с семейным доход свыше 50000 рублей.")

    income_order = ['Ученая степень', 'Два и более высших образования', 'Высшее', 'Неоконченное высшее', 'Среднее специальное', 'Среднее', 'Неполное среднее']
    data['Образование'] = pd.Categorical(data['Образование'], categories=income_order, ordered=True)
    contingency_table = pd.crosstab(data['Образование'], data['Целевая_переменная'])
    percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    plt.figure(figsize=(10, 6))
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
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Тепловая карта для семейного статуса (в процентах)', fontsize=15, pad=20)
    st.pyplot(plt.gcf())
    st.write("Наибольший отклик наблудается среди людей, состоящих в гражданском браке.")

    st.markdown("---")
    st.write("Поскольку тепловая корреляция должности и целевой переменной оказалась информативной, стоит дополнительно взглянуть на данные по отрасли работы клиента, а также по направлению деятельности клиента.")
    contingency_table = pd.crosstab(data['Отрасль_работы'], data['Целевая_переменная'])
    percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    plt.figure(figsize=(10, 20))
    sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Тепловая карта для отрасли работы клиента (в процентах)', fontsize=15, pad=20)
    st.pyplot(plt.gcf())
    st.write("При анализе корреляций между отраслями деятельности клиентов и откликом на услуги банка, можно выделить сферу недвижемости - 35% людей занятых в сфере недвижемости делают отклик на услуги банка. Следующий сектор - это сфера общественного питания и ресторанный бизнес.")

    contingency_table = pd.crosstab(data['Направление_деятельности'], data['Целевая_переменная'])
    percentage_distribution = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(percentage_distribution, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Тепловая карта для направления деятельности (в процентах)', fontsize=15, pad=20)
    st.pyplot(plt.gcf())
    st.write("Но если мы смотрим на деятельность самых клиентов, а не сферу занятости, то лидирующую позицию по откликам занимают люди, занимающиеся рекламой и маркетингом. Здесь стоит отметить, что на прошлом графиге маркетинг был с нулевыми показателями - здесь важно отличать отраслю от самого направления деятельности клиента. Подразумается, мне клиент может работать маркетологом, но в сфере недвижимости или общественного питания, но не в рекламной фирме.")
    st.markdown("---")
    st.subheader("Вывод")
    st.write()

    '''
    # Если выбран чекбокс для гистограмм, то выводим гистограммы для каждого признака
    if hist:
        st.header("Гистограммы распределений признаков")
        # Создаем список числовых признаков
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Для каждого числового признака строим гистограмму и выводим ее в приложении
        for col in num_cols:
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20)
            ax.set_title(f"Распределение {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Частота")
            st.pyplot(fig)
    '''


elif page == "Предсказательная модель":

    st.title("Предсказательная модель")
    # Выводим информацию о модели
    st.write("В этой странице мы используем модель ... для предсказания ... по введенным данным.")
    '''
    with st.form(key="my_form"):
        # Добавляем поля для ввода данных с помощью разных функций Streamlit
        # Например, st.number_input, st.text_input, st.select_slider, st.radio и т.д.
        # Присваиваем значения полей переменным
        # Например, x1 = st.number_input("Введите значение x1")
        # Добавляем кнопку для отправки формы с помощью функции st.form_submit_button
        submit_button = st.form_submit_button(label="Предсказать")
    # Если форма отправлена, то выполняем следующий код
    if submit_button:
        # Используем нашу модель для предсказания целевой переменной по введенным данным
        # Например, y_pred = model.predict([x1, x2, x3])
        # Выводим результат с помощью функции st.write или другой функции Streamlit
        # Например, st.write(f"Предсказанное значение y: {y_pred}")
        # Визуализируем результат с помощью библиотек matplotlib, seaborn, plotly и других
        # Например, st.pyplot, st.plotly_chart и т.д.
    '''