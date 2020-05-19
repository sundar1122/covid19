
import java.util.Scanner;
import java.util.Arrays;

public class BinarySearch {

   // return the index of the target in the sorted array data[]; 
   // return -1 if not found
   public static int search(String[] data, String target) {
      return search(data, target, 0, data.length);
   }
   
   /** Recursive binary search, case sensitive.
       @param data the array of strings to search
       @param target the word to search for
       @param lo the low end of the range to search, inclusive
       @param hi the high end of the range to search, exclusive
       @return the index of the target location, or -1 if not found
   */
   private static int search(String[] data, String target,
                               int lo, int hi) {
      // possible target indices in [lo, hi)
      // TODO: implement me!
      if (hi <= lo) {
         return -1;
      }
      int start = lo + (hi - lo) / 2;
      if (data[start].compareTo(target) == 0) {
         return start;
      }
      else if (data[start].compareTo(target) > 0) {
         return search(data, target, lo, start);
      }
      else if (data[start].compareTo(target) < 0) {
         return search(data, target, start + 1, hi);
      }
      
      return -1;
        
   }

   public static void main(String[] args) {
      final int N = 7;
      String[] a = new String[N];
      Scanner input = new Scanner(System.in);
      System.out.println("enter " + N + " input words");
      for (int i = 0; i < N; i++) {
         a[i] = input.next();
      }      
      Arrays.sort(a);
      System.out.println("Sorted array:");
      System.out.println(Arrays.toString(a));      
      
      String key;
      int where;
      System.out.print("enter keywords to search for, quit to end\n >> ");
      key = input.next();
   
      while (!"quit".equalsIgnoreCase(key)) { 
         where = search(a, key);
         if (where == -1) {
            System.out.println(key + " not found " );
         } else {
            System.out.println(key + " found at index " + where);
         }
         System.out.print(" >> ");
         key = input.next();
      }
   }
}
