Define a function called propagate_with_refinement.

propagate_with_refinement(x_ivl, u, scenarios).

For the first step, find $y_1=propagate_interval_euler(x_ivl, u, scenario1)$, $y_2=propagate_interval_euler(x_ivl, u, scenario1)$

Overlap between y_1 and y_2 is your first cost.

At the second step, find the interval defined by overlap between y_1 and y_2. Call this y(t_1). Find x(t_1) based on y(t_1): In the faulty car model, for nominal and actuator fault, y(t_1)'s entries are p_x, p_y which can be directly used to reconstruct the first two elements of x. The third element of x, theta, cannot be refined using knowledge of y. Leave it as is. For sensor fault, y(t_1)'s entries are p_x + 0.2, p_y + 0.2, so you can find out x interval easily by subtracting 0.2. Now you will get a different interval for x(t_1) for these two fault scenarios. Use them as x0_ivl to propagate your next controller.

Write this interval propagation function in faulty_car_separating_input.py. Then, define a loss function called refined_overlap_loss, similar to how you defined the original loss. Then write a gpu parallel optimization code that minimizes this loss.