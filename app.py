# -*- coding: utf-8 -*-
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import shinyswatch
from pathlib import Path

import function as fcts
import figures as figs
import numpy as np

from shinywidgets import output_widget, render_widget
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

css_path = Path(__file__).parent / "www" / "style.css"

app_ui = ui.page_fluid(
     ui.include_css(css_path),
    shinyswatch.theme.superhero(),
    ui.page_navbar(
        title=ui.div(ui.panel_title(ui.h2("Support Vector Machine (SVM) Explorer"),window_title="Support Vector Machine"),align="center"),
        inverse=True,id="navbar_id"
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.input_select(
                id="dropdown_select_dataset",
                label="Select Dataset:",
                choices={
                    "moons"   : "Moons",
                    "linear"  : "Linearly Separable",
                    "circles" : "Circles"
                },
                selected="moons",
                multiple=False
            ),
            ui.input_slider(
                id="slider_dataset_sample_size",
                label="Sample size:",
                min=100,
                max=500,
                step=100,
                value=300
            ),
            ui.input_slider(
                id="slider_dataset_noise_level",
                label="Noise Level:",
                min=0,
                max=1,
                step=0.1,
                value=0.2
            ),
            ui.input_slider(
                id="slider_threshold",
                label="Threshold:",
                min=0,
                max=1,
                step=0.01,
                value=0.5
            ),
            # ui.input_action_button(
            #     id="button_zero_threshold",
            #     label="Reset Threshold"
            # ),
            ui.input_select(
                id="dropdown_svm_parameter_kernel",
                label="Kernel:",
                choices={
                    "rbf"     : "Radial basis function (RBF)",
                    "linear"  : "Linear",
                    "poly"    : "Polynomial",
                    "sigmoid" : "Sigmoid"
                },
                selected="rbf",
                multiple=False
            ),
            ui.input_slider(
                id= "slider_svm_parameter_C_power",
                label="Cost (C):",
                min=-2,
                max=4,
                value=0
            ),
            ui.input_slider(
                    id="slider_svm_parameter_C_coef",
                    label="",
                    min=1,
                    max=9,
                    value=1
            ),
            ui.output_ui("Other_params"),
            ui.input_radio_buttons(
                id="radio_svm_parameter_shrinking",
                label="Shrinking :",
                choices={
                    True : "Enabled",
                    False : "Disabled"
                },
                selected=True,
                inline=True
            ),
        width="20%"
        ),
        ui.row(
            ui.column(2,
                fcts.card_performance(
                    id_name="accuracy_card",
                    perf_name="Accuracy",
                    perf_id="svm_accuracy_value"
                )
            ),
            ui.column(2,
                fcts.card_performance(
                    id_name="fscore_card",
                    perf_name="F1-score",
                    perf_id="svm_fscore_value"
                )
            ),
            ui.column(2,
                fcts.card_performance(
                    id_name="aucscore_card",
                    perf_name="AUC",
                    perf_id="svm_aucscore_value"
                )
            ),
            ui.column(2,
                fcts.card_performance(
                    id_name="precision_card",
                    perf_name="Precision",
                    perf_id="svm_precision_value"
                )
            ),
            ui.column(2,
                fcts.card_performance(
                    id_name="recall_card",
                    perf_name="Recall",
                    perf_id="svm_recall_value"
                )
            ),
            ui.column(2,
                fcts.card_performance(
                    id_name="averageprecision_card",
                    perf_name="Average Precision",
                    perf_id="svm_averageprecision_value"
                )
            )
        ),
        ui.row(
            ui.column(6, 
                output_widget("graph_sklearn_svm"),
                style = "background-color:#566573;"
            ),
            ui.column(3,
                ui.card(
                    ui.card_header("ROC Curve",style="text-align:center;"),
                    ui.output_plot("graph_roc_curve")
                )
            ),
            ui.column(3,
                ui.card(
                    ui.card_header("Confusion Matrix",style="text-align:center;"),
                    ui.output_plot("graph_confusion_pie")
                )
            )
        ),
        ui.row(
            ui.column(6,
                ui.card(
                    ui.card_header("Classification Report",style="text-align:center;"),
                    ui.div(ui.output_data_frame(id="svm_classification_report"),align="center")
                )
            )
        )
    )
)

def server(input,output,session):

    @output
    @render.ui
    def Other_params():
        if input.dropdown_svm_parameter_kernel() in ["rbf","sigmoid"]:
            return ui.TagList(
                        ui.input_slider(
                            id="slider_svm_parameter_gamma_power",
                            label="Gamma :",
                            min=-5,
                            max=0,
                            value=-1
                        ),
                        ui.input_slider(
                            id="slider_svm_parameter_gamma_coef",
                            label="",
                            min=1,
                            max=9,
                            value=5
                        )
                    )
        elif input.dropdown_svm_parameter_kernel() == "poly":
            return ui.TagList(
                        ui.input_slider(
                            id="slider_svm_parameter_degree",
                            label="Degree :",
                            min=2,
                            max=10,
                            value=3,
                            step=1
                        ),
                        ui.input_slider(
                            id="slider_svm_parameter_gamma_power",
                            label="Gamma :",
                            min=-5,
                            max=0,
                            value=-1
                        ),
                        ui.input_slider(
                            id="slider_svm_parameter_gamma_coef",
                            label="",
                            min=1,
                            max=9,
                            value=5
                        )
                    )
            
    # Create data
    @reactive.Calc
    def Model():
        h = 0.3  # step size in the mesh

        # Data Pre-processing
        X,y = fcts.generate_data(
            n_samples=input.slider_dataset_sample_size(),
            dataset=input.dropdown_select_dataset(),
            noise = input.slider_dataset_noise_level())
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min = X[:, 0].min() - 0.5
        x_max = X[:, 0].max() + 0.5
        y_min = X[:, 1].min() - 0.5
        y_max = X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        if input.dropdown_svm_parameter_kernel() in ["rbf","sigmoid"]:
            gamma_power = input.slider_svm_parameter_gamma_power()
            gamma_coef = input.slider_svm_parameter_gamma_coef()
            degree = 3
        elif input.dropdown_svm_parameter_kernel() == "poly":
            gamma_power = input.slider_svm_parameter_gamma_power()
            gamma_coef = input.slider_svm_parameter_gamma_coef()
            degree = input.slider_svm_parameter_degree()
        else:
            gamma_power = 0
            gamma_coef = 5
            degree = 3

        C = input.slider_svm_parameter_C_coef() * 10 ** input.slider_svm_parameter_C_power()
        gamma = gamma_coef * 10 ** gamma_power

        if input.radio_svm_parameter_shrinking():
            flag = True
        else:
            flag = False

        # Train SVM
        clf = SVC(C=C,
                  kernel=input.dropdown_svm_parameter_kernel(),
                  degree= degree,
                  gamma=gamma,
                  shrinking=flag,
                  probability=True)
        clf.fit(X_train, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        return clf, X_test, y_test, Z,X_train,y_train, xx, yy, h

    @output
    @render_widget
    def graph_sklearn_svm():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        Z = Model()[3]
        X_train = Model()[4]
        y_train = Model()[5]
        xx = Model()[6]
        yy = Model()[7]
        h = Model()[8]
        fig = figs.serve_prediction_plot(
            model=clf,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            Z= Z,
            xx=xx,
            yy=yy,
            mesh_step=h,
            threshold=input.slider_threshold()
        )
        return fig

    @output
    @render.plot()
    def graph_roc_curve():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        return figs.serve_roc_curve(
            model=clf,
            X_test=X_test,
            y_test=y_test
        )

    @output
    @render.plot()
    def graph_confusion_pie():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        Z = Model()[3]
        fig = figs.serve_confusion_matrix(
            model=clf,
            X_test=X_test,
            y_test=y_test,
            Z=Z,
            threshold=input.slider_threshold()
        )
        return fig

    # Return Accuracy
    @output
    @render.text
    def svm_accuracy_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.accuracy(
            model=clf,
            xtest=X_test,
            ytest=y_test,
            threshold=input.slider_threshold()
        )
        return f"{value}%"

    @output
    @render.text
    def svm_fscore_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.f1_score(
            model=clf,
            xtest=X_test,
            ytest=y_test,
            threshold=input.slider_threshold()
        )
        return f"{value}%"

    # Return AUC score
    @output
    @render.text
    def svm_aucscore_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.auc_roc_score(
            model=clf,
            xtest=X_test,
            ytest=y_test
        )
        return f"{value}%"

    @output
    @render.text
    def svm_precision_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.precision_score(
            model=clf,
            xtest=X_test,
            ytest=y_test,
            threshold=input.slider_threshold()
        )
        return f"{value}%"

    @output
    @render.text
    def svm_recall_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.recall_score(
            model=clf,
            xtest=X_test,
            ytest=y_test,
            threshold=input.slider_threshold()
        )
        return f"{value}%"

    @output
    @render.text
    def svm_averageprecision_value():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        value = fcts.average_precision_score(
            model=clf,
            xtest=X_test,
            ytest=y_test
        )
        return f"{value}%"
    
    # Classification report
    @output
    @render.data_frame
    def svm_classification_report():
        clf = Model()[0]
        X_test = Model()[1]
        y_test = Model()[2]
        data = fcts.classification_report(
            model=clf,
            xtest=X_test,
            ytest=y_test,
            threshold=input.slider_threshold()
        ).round(4).reset_index().rename(columns={"index":"Metrics"})
        return render.DataTable(data,filters=False,width="100%",row_selection_mode="multiple")

# Run App
app = App(app_ui,server)