import jax.numpy as jnp
import jax
import optax
import functools

@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
  return jnp.dot(params, x)

def compute_loss(params, x, y):
  y_pred = network(params, x)
  loss = jnp.mean(optax.l2_loss(y_pred, y))
  return loss

key = jax.random.PRNGKey(42)
target_params = 0.5

# Generate some data.
xs = jax.random.normal(key, (16, 2))
ys = jnp.sum(xs * target_params, axis=-1)

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
params = jnp.array([0.0, 0.0])
opt_state = optimizer.init(params)

# A simple update loop.
for _ in range(1000):
  grads = jax.grad(compute_loss)(params, xs, ys)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params), \
'Optimization should retrive the target params used to generate the data.'