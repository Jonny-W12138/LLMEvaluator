import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import importlib
import sys

st.header("Custom Report")

tabs = st.tabs(["Default template", "Custom"])
with tabs[0]:
    with st.container(border=True):
        st.write("Task")

        task = st.selectbox("Task name", os.listdir(os.path.join(os.getcwd(), "tasks")))
        if st.button("Select", key="select_task"):
            if task is not None or task != "":
                st.session_state["task"] = task

        if "task" not in st.session_state:
            st.warning("Choose a task to continue.")
            st.stop()

        if "task" in st.session_state:
            st.markdown(f"- Selected task: `{st.session_state['task']}`")

    with st.container(border=True):
        st.write("Ability")

        ability = st.selectbox("Ability name", [
            f for f in os.listdir(os.path.join(os.getcwd(), "src", "eval", "custom"))
            if os.path.isdir(os.path.join(os.getcwd(), "src", "eval", "custom", f))
        ])

        if st.button("Select", key="select_ability"):
            if ability is not None or ability != "":
                st.session_state["ability"] = ability

        if "ability" in st.session_state:
            st.markdown(f"- Selected ability: `{st.session_state["ability"]}`")
        else:
            st.markdown("- No ability selected.")

    if "ability" not in st.session_state:
        st.warning("Choose an ability to continue.")
        st.stop()

    if 'eval_json' not in st.session_state:
        st.session_state['eval_json'] = None
    if 'field_paths' not in st.session_state:
        st.session_state['field_paths'] = []

    with st.container(border=True):
        st.write("Evaluation")
        eval_file_path = st.text_input("Evaluation file path")

        if eval_file_path != "" :
            st.session_state['eval_file_path'] = eval_file_path

        if st.button("Select", key="select_eval"):
            if eval_file_path is not None or eval_file_path != "":
                with open(eval_file_path, "r") as eval_file:
                    eval_json = json.load(eval_file)
                st.session_state['eval_json'] = eval_json['evaluation']

    if st.session_state['eval_json'] is None:
        st.warning("No evaluation JSON file selected.")
        st.stop()

    def plot_line_chart(df, field, title, group_by=None):
        st.subheader(title)
        if group_by:
            fig = px.line(df, x=df.index, y=field, color=group_by, title=title)
        else:
            fig = px.line(df, x=df.index, y=field, title=title)
        st.plotly_chart(fig)

    def extract_field_value(item, field_path):
        fields = field_path.split('/')
        value = item
        for field in fields:
            if isinstance(value, dict) and field in value:
                value = value[field]
            else:
                return None
        return value

    def plot_bar_chart(df, fields, title, group_by=None):
        st.write(f"### {title}")

        # 确保 fields 是列表
        if not isinstance(fields, list):
            fields = [fields]

        valid_fields = [field for field in fields if field in df.columns]
        if not valid_fields:
            st.warning(f"None of the specified fields {fields} exist in the data.")
            return

        if group_by and group_by in df.columns:
            melted_df = df.melt(id_vars=[group_by], value_vars=valid_fields,
                                var_name="Metric", value_name="Value")
            count_df = melted_df.groupby([group_by, "Value"]).size().reset_index(name="Count")

            fig = px.bar(
                count_df,
                x=group_by,
                y="Count",
                color="Value",
                template="plotly_dark",
                barmode="group",
                text_auto=True,
                labels={group_by: "Category", "Count": "Count", "Value": "Value Type"}
            )


        else:
            melted_df = df.melt(value_vars=valid_fields, var_name="Metric", value_name="Value")
            fig = px.bar(
                melted_df,
                x="Metric",
                y="Value",
                template="plotly_dark",
                text_auto=True,
                labels={"Value": "Value", "Metric": "Category"}
            )

        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

    def plot_pie_chart(df, field, title, group_by=None):
        st.subheader(title)
        if group_by:
            unique_groups = df[group_by].unique()
            for group in unique_groups:
                group_df = df[df[group_by] == group]
                value_counts = group_df[field].value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]
                fig = px.pie(value_counts, values="Count", names="Value", title=f"{title} - {group}")
                st.plotly_chart(fig)
        else:
            value_counts = df[field].value_counts().reset_index()
            value_counts.columns = ["Value", "Count"]
            fig = px.pie(value_counts, values="Count", names="Value", title=title)
            st.plotly_chart(fig)

    def plot_radar_chart(df, field, title, group_by=None):
        st.subheader(title)

        if pd.api.types.is_numeric_dtype(df[field]):
            # For numeric fields, calculate the mean
            if group_by:
                grouped_data = df.groupby(group_by)[field].mean().reset_index()
                categories = grouped_data[group_by]
                values = grouped_data[field]
            else:
                values = [df[field].mean()]
                categories = ["Mean"]
        else:
            # For non-numeric fields, calculate the count
            if group_by:
                grouped_data = df.groupby([group_by, field]).size().reset_index(name="Count")
                categories = grouped_data[group_by]
                values = grouped_data["Count"]
            else:
                value_counts = df[field].value_counts().reset_index()
                categories = value_counts["index"]
                values = value_counts[field]

        fig = go.Figure()

        if group_by:
            for category in categories:
                fig.add_trace(go.Scatterpolar(
                    r=values[grouped_data[group_by] == category],
                    theta=categories[grouped_data[group_by] == category],
                    fill='toself',
                    name=category
                ))
        else:
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=title
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]  # Adjust range dynamically
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )

        st.plotly_chart(fig)


    def generate_eval_html_report(eval_json, field_paths_df):
        """Generate an HTML report for the evaluation data."""

        # Convert evaluation JSON to DataFrame
        results = []
        for item in eval_json:
            row = {}
            for _, row_data in field_paths_df.iterrows():
                field_path = row_data["Field Path"]
                group_by = row_data["Group By"]
                if field_path:
                    value = extract_field_value(item, field_path)
                    row[field_path] = value
                if group_by:
                    group_value = extract_field_value(item, group_by)
                    row[group_by] = group_value
            results.append(row)

        df = pd.DataFrame(results)

        # Generate charts based on field paths
        chart_htmls = []
        for _, row_data in field_paths_df.iterrows():
            field_path = row_data["Field Path"]
            chart_type = row_data["Chart Type"]
            chart_title = row_data["Chart Title"]
            group_by = row_data["Group By"]

            if field_path and field_path in df.columns:
                if chart_type == "Line Chart":
                    if pd.api.types.is_numeric_dtype(df[field_path]):
                        fig = px.line(df, x=df.index, y=field_path, color=group_by, template="plotly_white", title=chart_title)
                        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                        chart_htmls.append(f"<h3>{chart_title}</h3>{chart_html}")
                    else:
                        chart_htmls.append(
                            f"<p>Warning: The values for field path '{field_path}' are not numeric. Cannot generate a line chart.</p>")
                elif chart_type == "Bar Chart":
                    fields_to_plot = [row["Field Path"] for _, row in field_paths_df.iterrows() if row["Field Path"]]
                    if group_by and group_by in df.columns:
                        melted_df = df.melt(id_vars=[group_by], value_vars=fields_to_plot, var_name="Metric",
                                            value_name="Value")
                        count_df = melted_df.groupby([group_by, "Value"]).size().reset_index(name="Count")
                        fig = px.bar(count_df, x=group_by, y="Count", color="Value",  template="plotly_white",
                                     barmode="group", text_auto=True)
                    else:
                        melted_df = df.melt(value_vars=fields_to_plot, var_name="Metric", value_name="Value")
                        fig = px.bar(melted_df, x="Metric", y="Value", template="plotly_white", text_auto=True)
                    fig.update_traces(textposition='outside')
                    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                    chart_htmls.append(f"<h3>{chart_title}</h3>{chart_html}")
                elif chart_type == "Pie Chart":
                    if group_by:
                        unique_groups = df[group_by].unique()
                        for group in unique_groups:
                            group_df = df[df[group_by] == group]
                            value_counts = group_df[field_path].value_counts().reset_index()
                            value_counts.columns = ["Value", "Count"]
                            fig = px.pie(value_counts, values="Count", names="Value", template="plotly_white", title=f"{chart_title} - {group}")
                            chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                            chart_htmls.append(f"<h3>{chart_title} - {group}</h3>{chart_html}")
                    else:
                        value_counts = df[field_path].value_counts().reset_index()
                        value_counts.columns = ["Value", "Count"]
                        fig = px.pie(value_counts, values="Count", names="Value", template="plotly_white", title=chart_title)
                        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                        chart_htmls.append(f"<h3>{chart_title}</h3>{chart_html}")
                elif chart_type == "Radar Chart":
                    if pd.api.types.is_numeric_dtype(df[field_path]):
                        if group_by:
                            grouped_data = df.groupby(group_by)[field_path].mean().reset_index()
                            categories = grouped_data[group_by]
                            values = grouped_data[field_path]
                        else:
                            values = [df[field_path].mean()]
                            categories = ["Mean"]
                    else:
                        if group_by:
                            grouped_data = df.groupby([group_by, field_path]).size().reset_index(name="Count")
                            categories = grouped_data[group_by]
                            values = grouped_data["Count"]
                        else:
                            value_counts = df[field_path].value_counts().reset_index()
                            categories = value_counts["index"]
                            values = value_counts[field_path]

                    fig = go.Figure()
                    if group_by:
                        for category in categories:
                            fig.add_trace(go.Scatterpolar(
                                r=values[grouped_data[group_by] == category],
                                theta=categories[grouped_data[group_by] == category],
                                fill='toself',
                                name=category
                            ))
                    else:
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=chart_title
                        ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(values) * 1.1]
                            ),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        template="plotly_white",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=True
                    )
                    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                    chart_htmls.append(f"<h3>{chart_title}</h3>{chart_html}")
            else:
                chart_htmls.append(f"<p>Warning: Field path '{field_path}' not found.</p>")

        # Combine all chart HTMLs into a single report
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Evaluation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f4f4;
                }}
                .container {{
                    max-width: 1000px;
                    background: white;
                    padding: 20px;
                    margin: 0 auto;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                }}
                h1, h2, h3 {{
                    text-align: center;
                    color: #333;
                }}
                .chart-container {{
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluation Report</h1>
                {"".join([f"<div class='chart-container'>{chart_html}</div>" for chart_html in chart_htmls])}
            </div>
        </body>
        </html>
        """

        return html_template

    with st.container(border=True):
        st.write("Report charts")

        st.write("Edit Field Paths")

        if 'field_paths_df' not in st.session_state:
            st.session_state['field_paths_df'] = pd.DataFrame(columns=["Field Path", "Chart Type", "Chart Title", "Group By"])

        if st.session_state['field_paths_df'].empty:
            st.session_state['field_paths_df'] = pd.DataFrame([{"Field Path": "", "Chart Type": "Line Chart", "Chart Title": "", "Group By": ""}])

        edited_field_paths_df = st.data_editor(
            st.session_state['field_paths_df'],
            num_rows="dynamic",
            column_config={
                "Field Path": {"placeholder": "Enter field path (e.g., llm_output/translation)"},
                "Chart Type": st.column_config.SelectboxColumn(
                    "Chart Type",
                    help="Select the type of chart to generate",
                    options=["Line Chart", "Bar Chart", "Pie Chart", "Radar Chart"],
                    default="Line Chart",
                    required=True,
                ),
                "Chart Title": {"placeholder": "Enter chart title"},
                "Group By": {"placeholder": "Enter field path to group by (optional)"},
            },
            hide_index=True,
        )

        st.session_state['field_paths_df'] = edited_field_paths_df

        if not edited_field_paths_df.empty:
            results = []
            for item in st.session_state['eval_json']:
                row = {}
                for _, row_data in edited_field_paths_df.iterrows():
                    field_path = row_data["Field Path"]
                    group_by = row_data["Group By"]
                    if field_path:
                        value = extract_field_value(item, field_path)
                        row[field_path] = value
                    if group_by:
                        group_value = extract_field_value(item, group_by)
                        row[group_by] = group_value
                results.append(row)

            df = pd.DataFrame(results)

            for _, row_data in edited_field_paths_df.iterrows():
                field_path = row_data["Field Path"]
                chart_type = row_data["Chart Type"]
                chart_title = row_data["Chart Title"]
                group_by = row_data["Group By"]

                if field_path:
                    if field_path in df.columns:
                        if chart_type == "Line Chart":
                            if pd.api.types.is_numeric_dtype(df[field_path]):
                                plot_line_chart(df, field_path, chart_title, group_by)
                            else:
                                st.warning(f"The values for field path '{field_path}' are not numeric. Cannot generate a line chart.")
                        elif chart_type == "Bar Chart":
                            fields_to_plot = [row["Field Path"] for _, row in edited_field_paths_df.iterrows() if row["Field Path"]]
                            plot_bar_chart(df, fields_to_plot, chart_title, group_by)
                        elif chart_type == "Pie Chart":
                            plot_pie_chart(df, field_path, chart_title, group_by)
                        elif chart_type == "Radar Chart":
                            plot_radar_chart(df, field_path, chart_title, group_by)
                    else:
                        st.warning(f"Field path '{field_path}' not found.")
        else:
            st.info("Please enter at least one field path to generate charts.")

        if 'eval_json' in st.session_state and 'field_paths_df' in st.session_state:
            if st.download_button(
                    label="Download HTML Report",
                    data=generate_eval_html_report(st.session_state['eval_json'], st.session_state['field_paths_df']),
                    file_name="evaluation_report.html",
                    mime="text/html"
            ):
                st.success("HTML report downloaded successfully!")
        else:
            st.warning("No evaluation data or field paths found. Please load the data first.")

with tabs[1]:
    st.write("Custom report")

    script_path = st.text_input("Path to module(abs path)", key="custom_eval_script_path")
    function_name = st.text_input("Function name", key="custom_eval_function_name")

    if st.button("Generate", key="evaluate_custom_response"):
        script_path = os.path.abspath(script_path)

        # 获取模块名称（去掉.py）
        module_name = os.path.splitext(os.path.basename(script_path))[0]

        # 加载模块
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            raise ImportError(f"Could not load module from {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 获取函数
        if hasattr(module, function_name):
            function = getattr(module, function_name)
        else:
            raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'")

        with st.spinner("Executing ..."):
            function()
        st.success("Successfully generated.")