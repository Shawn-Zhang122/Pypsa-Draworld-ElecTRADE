fetch("../data/snapshots/latest.json")
  .then(r => r.json())
  .then(data => {
    console.log("Loaded snapshot:", data);
  });
