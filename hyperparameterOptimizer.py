# allegro trains integration
    # automation modules for hyperparameter optimization
from trains.automation import UniformParameterRange, UniformIntegerParameterRange
from trains.automation import HyperParameterOptimizer
from trains.automation.optuna import OptimizerOptuna

#### optimize hyperparameters
from trains import Task
task = Task.create(project_name='agGRU', 
                task_name='agGRU-Check Hyperparameter optimization')

optimizer = HyperParameterOptimizer(
    base_task_id='f7c66251e33049999fc17ec0d8cacc32',  
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=20, step_size=2),
        UniformIntegerParameterRange('dimension', min_value=16, max_value=480, step_size=2),
        UniformIntegerParameterRange('sequence feature depth', min_value=480, max_value=800, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=64, step_size=2),
        UniformParameterRange('dropout_output', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('dropout_between_layers', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('learning_rate', min_value=0.00001, max_value=0.01, step_size=0.000015),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='total_testing_loss',
    objective_metric_series='total_testing_loss',
    objective_metric_sign='min',  

    # setting optimizer 
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='hyperparameter_optimization',  
    max_number_of_concurrent_tasks=2,  
    optimization_time_limit=60., 
    compute_time_limit=120, 
    total_max_jobs=20,  
    min_iteration_per_job=15000,  
    max_iteration_per_job=150000,  
)

optimizer.set_report_period(1) # setting the time gap between two consecutive reports
optimizer.start()
optimizer.wait() # wait until process is done
optimizer.stop() # make sure background optimization stopped

