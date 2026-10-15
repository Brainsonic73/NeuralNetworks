# %% [markdown]
# ## Part 3: Hyperparameter Optimisation
#
# Grid search over three hyperparameters:
# - Learning rate: {0.001, 0.01, 0.1}
# - Batch size: {32, 64, 128}
# - Dropout: {0.0, 0.2, 0.5}
#
# That's 27 combinations, each trained for 10 epochs.
#
# **Note:** If `part3_results.csv` exists, we load the cached results instead of
# re-running the (95-minute) grid search. This keeps notebook re-runs fast.

# %%
if os.path.exists('part3_results.csv') and os.path.exists('part3_histories.json'):
    print("Loading cached results from part3_results.csv")
    df = pd.read_csv('part3_results.csv')
    with open('part3_histories.json') as f:
        all_histories = json.load(f)
    print(f"Loaded {len(df)} runs from cache")
else:
    print("No cache found — running full grid search (~95 min)")
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    dropouts = [0.0, 0.2, 0.5]
    results = []
    all_histories = {}
    total_runs = len(learning_rates) * len(batch_sizes) * len(dropouts)
    run_id = 0
    start_time = time.time()

    for lr in learning_rates:
        for bs in batch_sizes:
            for do in dropouts:
                run_id += 1
                key = f"lr{lr}_bs{bs}_do{do}"
                print(f"\n[{run_id}/{total_runs}] {key}")

                train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

                torch.manual_seed(42)
                model = MLP(dropout=do)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr)

                t0 = time.time()
                hist = train_model(model, train_loader, val_loader,
                                   criterion, optimizer, epochs=10)
                elapsed = time.time() - t0

                print(f"   Final val acc: {hist['val_acc'][-1]:.4f}  "
                      f"(best: {max(hist['val_acc']):.4f})  Time: {elapsed:.1f}s")

                results.append({
                    'lr': lr, 'batch_size': bs, 'dropout': do,
                    'final_val_acc': hist['val_acc'][-1],
                    'best_val_acc': max(hist['val_acc']),
                    'train_time_s': elapsed
                })
                all_histories[key] = hist

    df = pd.DataFrame(results)
    df.to_csv('part3_results.csv', index=False)
    with open('part3_histories.json', 'w') as f:
        json.dump(all_histories, f)
    print(f"\nTotal time: {(time.time() - start_time)/60:.1f} min")
    print(df.to_string(index=False))

# %%
# Show the top-performing configurations
print("\n=== Results sorted by best val acc ===")
print(df.sort_values('best_val_acc', ascending=False).head(10).to_string(index=False))

best_row = df.loc[df['best_val_acc'].idxmax()]
print(f"\nBest observed configuration:")
print(f"  lr         = {best_row['lr']}")
print(f"  batch_size = {best_row['batch_size']}")
print(f"  dropout    = {best_row['dropout']}")
print(f"  Best val acc = {best_row['best_val_acc']:.4f}")
