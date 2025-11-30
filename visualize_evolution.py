#!/usr/bin/env python3
"""
Evolution Animation Script
==========================
Creates an animated GIF showing how the neural network evolves
over generations. Perfect for understanding how NEAT adds nodes
and connections to solve increasingly complex problems.
"""

import os
import sys
from glob import glob
from PIL import Image


def create_evolution_gif(run_dir, duration=500, loop=0):
    """
    Create an animated GIF from network visualizations.

    Args:
        run_dir: Directory containing the training run
        duration: Duration of each frame in milliseconds
        loop: Number of times to loop (0 = infinite)
    """

    # Support both old and new directory structures
    networks_dir = os.path.join(run_dir, 'visualizations', 'networks')
    if not os.path.exists(networks_dir):
        networks_dir = os.path.join(run_dir, 'networks')

    if not os.path.exists(networks_dir):
        print(f"\n❌ Error: Networks directory not found: {networks_dir}")
        print("💡 Run visualize_network.py first to generate network diagrams.\n")
        return

    # Find all network images
    image_files = sorted(glob(os.path.join(networks_dir, 'net_g*.png')))

    if not image_files:
        print(f"\n❌ Error: No network images found in {networks_dir}")
        print("💡 Run visualize_network.py first to generate network diagrams.\n")
        return

    print("\n" + "="*70)
    print("🎬 CREATING EVOLUTION ANIMATION")
    print("="*70)
    print(f"📁 Run directory: {run_dir}")
    print(f"🖼️  Found {len(image_files)} network images")
    print(f"⏱️  Frame duration: {duration}ms")
    print("="*70 + "\n")

    # Load images
    print("📸 Loading images...")
    images = []
    for img_file in image_files:
        img = Image.open(img_file)
        images.append(img)
        gen_num = img_file.split('_g')[-1].split('.png')[0]
        print(f"  ✓ Loaded generation {gen_num}")

    # Create GIF in visualizations directory
    viz_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_file = os.path.join(viz_dir, 'neat_evolution.gif')

    print(f"\n🎨 Creating animated GIF...")

    # Save as animated GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print("\n" + "="*70)
    print("✅ ANIMATION COMPLETE!")
    print("="*70)
    print(f"🎬 Saved to: {output_file}")
    print(f"📦 File size: {file_size_mb:.2f} MB")
    print(f"🖼️  Frames: {len(images)}")
    print(f"⏱️  Duration per frame: {duration}ms")
    print(f"🔁 Loop: {'infinite' if loop == 0 else f'{loop} times'}")
    print("="*70)
    print("\n💡 You can now:")
    print(f"   - Open the GIF in any image viewer")
    print(f"   - Share it to show how NEAT evolves networks!")
    print("="*70 + "\n")


def create_side_by_side_comparison(run_dir, generations_to_compare=None):
    """
    Create a side-by-side comparison of networks from different generations.

    Args:
        run_dir: Directory containing the training run
        generations_to_compare: List of generation numbers to compare
                               If None, compares first, middle, and last
    """

    # Support both old and new directory structures
    networks_dir = os.path.join(run_dir, 'visualizations', 'networks')
    if not os.path.exists(networks_dir):
        networks_dir = os.path.join(run_dir, 'networks')

    if not os.path.exists(networks_dir):
        print(f"\n❌ Error: Networks directory not found: {networks_dir}")
        return

    image_files = sorted(glob(os.path.join(networks_dir, 'net_g*.png')))

    if not image_files:
        print(f"\n❌ Error: No network images found in {networks_dir}")
        return

    # Determine which generations to compare
    if generations_to_compare is None:
        num_images = len(image_files)
        indices = [0, num_images // 2, num_images - 1]
        selected_files = [image_files[i] for i in indices]
    else:
        selected_files = [
            os.path.join(networks_dir, f'net_g{gen:03d}.png')
            for gen in generations_to_compare
        ]
        selected_files = [f for f in selected_files if os.path.exists(f)]

    if len(selected_files) < 2:
        print("❌ Need at least 2 networks to compare")
        return

    print("\n" + "="*70)
    print("🔍 CREATING SIDE-BY-SIDE COMPARISON")
    print("="*70)
    print(f"🖼️  Comparing {len(selected_files)} generations")
    print("="*70 + "\n")

    # Load images
    images = [Image.open(f) for f in selected_files]

    # Calculate dimensions for side-by-side layout
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new image
    comparison = Image.new('RGB', (total_width, max_height), 'white')

    # Paste images side by side
    x_offset = 0
    for img in images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save comparison in visualizations directory
    viz_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_file = os.path.join(viz_dir, 'network_comparison.png')
    comparison.save(output_file)

    print(f"✅ Saved comparison to: {output_file}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python visualize_evolution.py <run_directory> [options]")
        print("\nExample:")
        print("  python visualize_evolution.py runs/run_20241126_120000")
        print("\nOptions:")
        print("  --duration <ms>     Frame duration in milliseconds (default: 500)")
        print("  --compare           Also create side-by-side comparison")
        print("  --fast              Use 200ms frame duration")
        print("  --slow              Use 1000ms frame duration")
        print()
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    # Parse options
    duration = 500  # default

    if '--fast' in sys.argv:
        duration = 200
    elif '--slow' in sys.argv:
        duration = 1000
    elif '--duration' in sys.argv:
        try:
            idx = sys.argv.index('--duration')
            duration = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Invalid duration value")
            sys.exit(1)

    # Create GIF
    create_evolution_gif(run_dir, duration=duration)

    # Create comparison if requested
    if '--compare' in sys.argv:
        create_side_by_side_comparison(run_dir)
