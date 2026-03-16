import taichi as ti
import numpy as np
from simulation import FluidSimulation

def test_dye_gradient_force():
    res = 128
    sim = FluidSimulation(res)
    
    # Initialize with a simple pattern: a Gaussian blob in the center
    @ti.kernel
    def init_gaussian():
        for i, j in sim.rho:
            dist2 = (i/res - 0.5)**2 + (j/res - 0.5)**2
            sim.rho[i, j] = ti.exp(-dist2 * 100)
            
    init_gaussian()
    
    # Check initial velocity is zero
    vel_np = sim.vel.to_numpy()
    assert np.all(vel_np == 0)
    
    # Apply dye gradient force
    scale = 100.0
    duration = 0.1
    sim.apply_dye_gradient_force(scale=scale, duration=duration)
    
    # Run one step
    sim.step()
    
    # Check velocity field
    vel_np = sim.vel.to_numpy()
    
    # The gradient of a Gaussian exp(-r^2) is -2r * exp(-r^2)
    # So the force (gradient) points towards the center if rho is exp(-r^2)
    # Wait, the gradient points UP the slope. So if rho = exp(-r^2), gradient points TOWARDS the center.
    # Actually, grad(exp(-r^2)) = -2r * exp(-r^2). 
    # If r is the vector from center to point, then -r points towards center.
    # So velocity should be non-zero and generally pointing towards the center.
    
    magnitude = np.sqrt(vel_np[:, :, 0]**2 + vel_np[:, :, 1]**2)
    max_mag = np.max(magnitude)
    print(f"Max velocity magnitude after one step: {max_mag}")
    
    assert max_mag > 0, "Velocity should be non-zero after applying force"
    
    # Check if the force is dynamic: change dye and see if gradient changes in the next step
    # Let's shift the dye
    @ti.kernel
    def shift_dye():
        for i, j in sim.rho:
            sim.rho[i, j] = 0.0
        for i, j in sim.rho:
            dist2 = (i/res - 0.7)**2 + (j/res - 0.7)**2
            sim.rho[i, j] = ti.exp(-dist2 * 100)
            
    shift_dye()
    
    # Store current velocity
    old_vel = vel_np.copy()
    
    # Step again
    sim.step()
    
    new_vel = sim.vel.to_numpy()
    delta_vel = new_vel - old_vel
    
    delta_mag = np.sqrt(delta_vel[:, :, 0]**2 + delta_vel[:, :, 1]**2)
    max_delta_mag = np.max(delta_mag)
    print(f"Max delta velocity magnitude after shifting dye: {max_delta_mag}")
    
    assert max_delta_mag > 0, "Force should have changed after dye was shifted"

    print("Dynamic dye gradient force verification successful!")

if __name__ == "__main__":
    test_dye_gradient_force()
