dt = 0.05
hover_time = 10
num_steps = int(hover_time / dt)
dry_mass = 50
fuel_mass = 26.73624842
total_impulse = 0

for step in range(num_steps):
    current_time = step * dt
    thrust_required = (dry_mass + fuel_mass) * 9.81
    total_impulse += thrust_required * dt
    fuel_mass -= (1/(9.81 * 180)) * thrust_required * dt
    print(f"Time: {current_time:.2f} s, Thrust Required: {thrust_required:.2f} N")

print(f"Total Impulse over {hover_time} seconds: {total_impulse:.2f} Ns")
print(f"Remaining Fuel Mass after {hover_time} seconds: {fuel_mass:.2f} kg")
print(f"Mass Remaining: {dry_mass + fuel_mass:.2f} kg")